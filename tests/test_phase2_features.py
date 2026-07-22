from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import yaml

from src.data import (
    ImagePreprocessor,
    ManifestImageDataset,
    load_manifest,
    records_for_split,
)
from src.features import (
    CACHE_ARRAY_NAMES,
    HandcraftedFeatureConfig,
    build_split_feature_cache,
    extract_handcrafted_features,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_features.yaml"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "prototype"
    / "selection_manifest.csv"
)


def load_feature_config() -> (
    HandcraftedFeatureConfig
):
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return HandcraftedFeatureConfig.from_mapping(
        config["feature_contract"]
    )


def build_sample_dataset(
    count: int = 4,
) -> ManifestImageDataset:
    records = load_manifest(
        MANIFEST_PATH
    )

    train_records = records_for_split(
        records,
        "train",
    )[:count]

    return ManifestImageDataset(
        records=train_records,
        project_root=PROJECT_ROOT,
        preprocessor=ImagePreprocessor(
            width=224,
            height=224,
            colour_mode="RGB",
        ),
        output_mode="numpy",
    )


def test_feature_dimensions_and_lbp_is_warning_free() -> None:
    config = load_feature_config()
    dataset = build_sample_dataset(2)

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter("always")

        bundles = [
            extract_handcrafted_features(
                dataset[index]["image"],
                config,
            )
            for index in range(2)
        ]

    lbp_warnings = [
        warning
        for warning in caught
        if "local_binary_pattern"
        in str(warning.message)
    ]

    assert not lbp_warnings

    for bundle in bundles:
        vectors = bundle.as_dict()

        assert vectors["hog"].shape == (26244,)
        assert vectors["lbp"].shape == (26,)
        assert vectors["fft"].shape == (32,)
        assert vectors["dct"].shape == (255,)
        assert vectors["combined"].shape == (
            26557,
        )

        for vector in vectors.values():
            assert vector.dtype == np.float32
            assert np.isfinite(vector).all()


def test_feature_extraction_is_deterministic() -> None:
    config = load_feature_config()
    dataset = build_sample_dataset(1)
    image = dataset[0]["image"]

    first = extract_handcrafted_features(
        image,
        config,
    )

    second = extract_handcrafted_features(
        image,
        config,
    )

    for name in first.as_dict():
        assert np.array_equal(
            first.as_dict()[name],
            second.as_dict()[name],
        )


def test_small_feature_cache_contract(
    tmp_path: Path,
) -> None:
    config = load_feature_config()
    records = load_manifest(
        MANIFEST_PATH
    )

    selected_records = records_for_split(
        records,
        "train",
    )[:4]

    metadata = build_split_feature_cache(
        records=selected_records,
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
        preprocessor=ImagePreprocessor(),
        feature_config=config,
        split="train",
        show_progress=False,
    )

    assert metadata["record_count"] == 4

    expected_shapes = {
        "hog": (4, 26244),
        "lbp": (4, 26),
        "fft": (4, 32),
        "dct": (4, 255),
        "combined": (4, 26557),
        "labels": (4,),
        "image_ids": (4,),
    }

    for name in CACHE_ARRAY_NAMES:
        path = (
            tmp_path
            / "train"
            / f"{name}.npy"
        )

        assert path.is_file()
        assert (
            metadata["files"][name]["path"]
            == f"train/{name}.npy"
        )

        array = np.load(
            path,
            allow_pickle=False,
        )

        assert array.shape == expected_shapes[name]

        assert (
            sha256_file(path)
            == metadata["files"][name]["sha256"]
        )


def test_small_cache_files_are_reproducible(
    tmp_path: Path,
) -> None:
    config = load_feature_config()
    records = load_manifest(
        MANIFEST_PATH
    )

    selected_records = records_for_split(
        records,
        "train",
    )[:4]

    roots = [
        tmp_path / "first",
        tmp_path / "second",
    ]

    for root in roots:
        build_split_feature_cache(
            records=selected_records,
            project_root=PROJECT_ROOT,
            output_root=root,
            preprocessor=ImagePreprocessor(),
            feature_config=config,
            split="train",
            show_progress=False,
        )

    for name in CACHE_ARRAY_NAMES:
        first_path = (
            roots[0]
            / "train"
            / f"{name}.npy"
        )

        second_path = (
            roots[1]
            / "train"
            / f"{name}.npy"
        )

        assert (
            sha256_file(first_path)
            == sha256_file(second_path)
        )
