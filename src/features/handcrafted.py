from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from scipy.fft import dctn, fft2, fftshift
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern


FEATURE_ORDER = (
    "hog",
    "lbp",
    "fft",
    "dct",
)


@dataclass(frozen=True, slots=True)
class HandcraftedFeatureConfig:
    width: int
    height: int
    colour_mode: str
    value_minimum: float
    value_maximum: float

    hog_orientations: int
    hog_pixels_per_cell: tuple[int, int]
    hog_cells_per_block: tuple[int, int]
    hog_block_norm: str
    hog_dimension: int

    lbp_points: int
    lbp_radius: float
    lbp_method: str
    lbp_bins: int
    lbp_dimension: int

    fft_bins: int
    fft_use_log_magnitude: bool
    fft_dimension: int

    dct_block_size: int
    dct_exclude_dc: bool
    dct_dimension: int

    combined_order: tuple[str, ...]
    combined_dimension: int

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> HandcraftedFeatureConfig:
        input_config = mapping["input"]
        hog_config = mapping["hog"]
        lbp_config = mapping["lbp"]
        fft_config = mapping["fft"]
        dct_config = mapping["dct"]
        combined_config = mapping["combined"]

        config = cls(
            width=int(input_config["width"]),
            height=int(input_config["height"]),
            colour_mode=str(input_config["colour_mode"]),
            value_minimum=float(
                input_config["value_minimum"]
            ),
            value_maximum=float(
                input_config["value_maximum"]
            ),
            hog_orientations=int(
                hog_config["orientations"]
            ),
            hog_pixels_per_cell=tuple(
                int(value)
                for value in hog_config[
                    "pixels_per_cell"
                ]
            ),
            hog_cells_per_block=tuple(
                int(value)
                for value in hog_config[
                    "cells_per_block"
                ]
            ),
            hog_block_norm=str(
                hog_config["block_norm"]
            ),
            hog_dimension=int(
                hog_config["dimension"]
            ),
            lbp_points=int(lbp_config["points"]),
            lbp_radius=float(lbp_config["radius"]),
            lbp_method=str(lbp_config["method"]),
            lbp_bins=int(lbp_config["bins"]),
            lbp_dimension=int(
                lbp_config["dimension"]
            ),
            fft_bins=int(fft_config["bins"]),
            fft_use_log_magnitude=bool(
                fft_config["use_log_magnitude"]
            ),
            fft_dimension=int(
                fft_config["dimension"]
            ),
            dct_block_size=int(
                dct_config["block_size"]
            ),
            dct_exclude_dc=bool(
                dct_config["exclude_dc"]
            ),
            dct_dimension=int(
                dct_config["dimension"]
            ),
            combined_order=tuple(
                str(value)
                for value in combined_config["order"]
            ),
            combined_dimension=int(
                combined_config["dimension"]
            ),
        )

        config.validate()
        return config

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "hog": self.hog_dimension,
            "lbp": self.lbp_dimension,
            "fft": self.fft_dimension,
            "dct": self.dct_dimension,
            "combined": self.combined_dimension,
        }

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Feature input dimensions must be positive"
            )

        if self.colour_mode != "RGB":
            raise ValueError(
                "Handcrafted feature input must be RGB"
            )

        if self.value_minimum >= self.value_maximum:
            raise ValueError(
                "Invalid feature input value range"
            )

        if len(self.hog_pixels_per_cell) != 2:
            raise ValueError(
                "HOG pixels_per_cell must have two values"
            )

        if len(self.hog_cells_per_block) != 2:
            raise ValueError(
                "HOG cells_per_block must have two values"
            )

        expected_lbp_bins = self.lbp_points + 2

        if self.lbp_bins != expected_lbp_bins:
            raise ValueError(
                "Uniform LBP requires points + 2 bins"
            )

        if self.lbp_dimension != self.lbp_bins:
            raise ValueError(
                "LBP dimension must equal its bin count"
            )

        expected_dct_dimension = (
            self.dct_block_size ** 2
            - int(self.dct_exclude_dc)
        )

        if self.dct_dimension != expected_dct_dimension:
            raise ValueError(
                "Configured DCT dimension is inconsistent"
            )

        if self.combined_order != FEATURE_ORDER:
            raise ValueError(
                "Unsupported combined feature order"
            )

        expected_combined_dimension = sum(
            self.dimensions[name]
            for name in FEATURE_ORDER
        )

        if (
            self.combined_dimension
            != expected_combined_dimension
        ):
            raise ValueError(
                "Configured combined dimension is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    hog: np.ndarray
    lbp: np.ndarray
    fft: np.ndarray
    dct: np.ndarray
    combined: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "hog": self.hog,
            "lbp": self.lbp,
            "fft": self.fft,
            "dct": self.dct,
            "combined": self.combined,
        }


def _l2_normalise(
    vector: np.ndarray,
) -> np.ndarray:
    result = np.asarray(
        vector,
        dtype=np.float32,
    ).copy()

    norm = float(np.linalg.norm(result))

    if norm > 0.0:
        result /= norm

    return result


def _validate_image(
    image: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> np.ndarray:
    array = np.asarray(image)

    expected_shape = (
        config.height,
        config.width,
        3,
    )

    if array.shape != expected_shape:
        raise ValueError(
            f"Expected image shape {expected_shape}, "
            f"found {array.shape}"
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(
            "Feature input must be numeric"
        )

    array = array.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(array).all():
        raise ValueError(
            "Feature input contains non-finite values"
        )

    tolerance = 1e-6

    if (
        float(array.min())
        < config.value_minimum - tolerance
        or float(array.max())
        > config.value_maximum + tolerance
    ):
        raise ValueError(
            "Feature input lies outside the configured range"
        )

    return array


def _grayscale_float(
    image: np.ndarray,
) -> np.ndarray:
    return rgb2gray(image).astype(
        np.float32,
        copy=False,
    )


def _grayscale_uint8(
    grayscale: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        grayscale,
        0.0,
        1.0,
    )

    return np.rint(
        clipped * 255.0
    ).astype(np.uint8)


@lru_cache(maxsize=16)
def _radial_bin_map(
    height: int,
    width: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    centre_y = (height - 1) / 2.0
    centre_x = (width - 1) / 2.0

    y_coordinates, x_coordinates = np.indices(
        (height, width),
        dtype=np.float64,
    )

    radius = np.sqrt(
        (x_coordinates - centre_x) ** 2
        + (y_coordinates - centre_y) ** 2
    )

    maximum_radius = float(radius.max())

    if maximum_radius <= 0.0:
        raise ValueError(
            "Cannot calculate radial bins for this image"
        )

    bin_indices = np.floor(
        radius / maximum_radius * bins
    ).astype(np.int64)

    bin_indices = np.clip(
        bin_indices,
        0,
        bins - 1,
    ).ravel()

    counts = np.bincount(
        bin_indices,
        minlength=bins,
    ).astype(np.float64)

    bin_indices.setflags(write=False)
    counts.setflags(write=False)

    return bin_indices, counts


def extract_hog(
    grayscale: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> np.ndarray:
    vector = hog(
        grayscale,
        orientations=config.hog_orientations,
        pixels_per_cell=config.hog_pixels_per_cell,
        cells_per_block=config.hog_cells_per_block,
        block_norm=config.hog_block_norm,
        feature_vector=True,
    ).astype(np.float32)

    return vector


def extract_lbp(
    grayscale_uint8: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> np.ndarray:
    lbp_image = local_binary_pattern(
        grayscale_uint8,
        P=config.lbp_points,
        R=config.lbp_radius,
        method=config.lbp_method,
    )

    bin_edges = np.arange(
        config.lbp_bins + 1,
        dtype=np.float64,
    )

    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=bin_edges,
    )

    histogram = histogram.astype(
        np.float32
    )

    total = float(histogram.sum())

    if total > 0.0:
        histogram /= total

    return histogram


def extract_fft_radial_profile(
    grayscale: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> np.ndarray:
    spectrum = fftshift(
        fft2(grayscale)
    )

    magnitude = np.abs(spectrum)

    if config.fft_use_log_magnitude:
        magnitude = np.log1p(magnitude)

    bin_indices, counts = _radial_bin_map(
        grayscale.shape[0],
        grayscale.shape[1],
        config.fft_bins,
    )

    weighted_sums = np.bincount(
        bin_indices,
        weights=magnitude.ravel(),
        minlength=config.fft_bins,
    )

    profile = np.divide(
        weighted_sums,
        counts,
        out=np.zeros_like(
            weighted_sums,
            dtype=np.float64,
        ),
        where=counts > 0,
    )

    return _l2_normalise(profile)


def extract_dct_low_frequency(
    grayscale: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> np.ndarray:
    coefficients = dctn(
        grayscale,
        type=2,
        norm="ortho",
    )

    block = coefficients[
        : config.dct_block_size,
        : config.dct_block_size,
    ].astype(
        np.float32,
        copy=True,
    )

    vector = block.ravel()

    if config.dct_exclude_dc:
        vector = vector[1:]

    return _l2_normalise(vector)


def extract_handcrafted_features(
    image: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> FeatureBundle:
    validated_image = _validate_image(
        image,
        config,
    )

    grayscale = _grayscale_float(
        validated_image
    )

    grayscale_uint8 = _grayscale_uint8(
        grayscale
    )

    vectors = {
        "hog": extract_hog(
            grayscale,
            config,
        ),
        "lbp": extract_lbp(
            grayscale_uint8,
            config,
        ),
        "fft": extract_fft_radial_profile(
            grayscale,
            config,
        ),
        "dct": extract_dct_low_frequency(
            grayscale,
            config,
        ),
    }

    for name, vector in vectors.items():
        expected_dimension = config.dimensions[name]

        if vector.shape != (
            expected_dimension,
        ):
            raise RuntimeError(
                f"{name} produced {vector.shape}; "
                f"expected {(expected_dimension,)}"
            )

        if vector.dtype != np.float32:
            raise RuntimeError(
                f"{name} did not produce float32"
            )

        if not np.isfinite(vector).all():
            raise RuntimeError(
                f"{name} produced non-finite values"
            )

    combined = np.concatenate(
        [
            vectors[name]
            for name in config.combined_order
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    if combined.shape != (
        config.combined_dimension,
    ):
        raise RuntimeError(
            "Combined feature dimension is incorrect"
        )

    return FeatureBundle(
        hog=vectors["hog"],
        lbp=vectors["lbp"],
        fft=vectors["fft"],
        dct=vectors["dct"],
        combined=combined,
    )
