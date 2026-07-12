from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as vision_functional


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STANDARD_DEVIATION = (
    0.229,
    0.224,
    0.225,
)


class ImageDecodingError(RuntimeError):
    """Raised when an image cannot be safely decoded."""


@dataclass(frozen=True, slots=True)
class ImagePreprocessor:
    width: int = 224
    height: int = 224
    colour_mode: str = "RGB"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Image dimensions must be positive"
            )

        if self.colour_mode != "RGB":
            raise ValueError(
                "The prototype preprocessing contract "
                "requires RGB output"
            )

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def process_pil(
        self,
        image: Image.Image,
    ) -> Image.Image:
        converted = image.convert(self.colour_mode)

        resized = converted.resize(
            self.size,
            resample=Image.Resampling.BILINEAR,
        )

        return resized

    def load(
        self,
        image_path: Path,
    ) -> Image.Image:
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        try:
            with Image.open(image_path) as image:
                image.load()
                processed = self.process_pil(image)
                return processed.copy()

        except FileNotFoundError:
            raise

        except Exception as exc:
            raise ImageDecodingError(
                f"Could not decode image: {image_path}"
            ) from exc

    def to_numpy(
        self,
        image_path: Path,
        *,
        scale_to_unit_interval: bool = True,
    ) -> np.ndarray:
        image = self.load(image_path)

        array = np.asarray(image).copy()

        if scale_to_unit_interval:
            array = array.astype(
                np.float32,
                copy=False,
            )

            array /= 255.0

        return array

    def to_tensor(
        self,
        image_path: Path,
        *,
        normalise_for_imagenet: bool = True,
    ) -> torch.Tensor:
        image = self.load(image_path)

        tensor = vision_functional.pil_to_tensor(
            image
        ).to(dtype=torch.float32)

        tensor /= 255.0

        if normalise_for_imagenet:
            tensor = vision_functional.normalize(
                tensor,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STANDARD_DEVIATION,
            )

        return tensor
