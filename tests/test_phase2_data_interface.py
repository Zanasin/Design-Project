from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data import (
    ImagePreprocessor,
    build_development_dataset,
    build_evaluation_dataset,
    count_by_split_and_label,
    load_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "prototype"
    / "selection_manifest.csv"
)

PROTOTYPE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype.yaml"
)


@pytest.fixture(scope="module")
def preprocessor() -> ImagePreprocessor:
    with PROTOTYPE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return ImagePreprocessor(
        width=int(config["image"]["width"]),
        height=int(config["image"]["height"]),
        colour_mode=config["image"]["colour_mode"],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def test_manifest_loader_counts_and_contract() -> None:
    records = load_manifest(MANIFEST_PATH)

    assert len(records) == 1000

    expected_counts = Counter(
        {
            ("train", "real"): 350,
            ("train", "ai_generated"): 350,
            ("validation", "real"): 75,
            ("validation", "ai_generated"): 75,
            ("test", "real"): 75,
            ("test", "ai_generated"): 75,
        }
    )

    assert (
        count_by_split_and_label(records)
        == expected_counts
    )

    assert len(
        {record.image_id for record in records}
    ) == 1000

    assert len(
        {
            record.selected_relative_path
            for record in records
        }
    ) == 1000

    assert len(
        {record.source_sha256 for record in records}
    ) == 1000


def test_development_builder_rejects_test_split(
    preprocessor: ImagePreprocessor,
) -> None:
    with pytest.raises(
        ValueError,
        match="evaluation-only",
    ):
        build_development_dataset(
            manifest_path=MANIFEST_PATH,
            project_root=PROJECT_ROOT,
            preprocessor=preprocessor,
            split="test",
            output_mode="tensor",
        )


def test_preprocessor_outputs_are_deterministic(
    preprocessor: ImagePreprocessor,
) -> None:
    records = load_manifest(MANIFEST_PATH)
    image_path = records[0].absolute_path(
        PROJECT_ROOT
    )

    pil_image = preprocessor.load(image_path)

    assert isinstance(pil_image, Image.Image)
    assert pil_image.mode == "RGB"
    assert pil_image.size == (224, 224)

    numpy_image_1 = preprocessor.to_numpy(
        image_path
    )

    numpy_image_2 = preprocessor.to_numpy(
        image_path
    )

    assert numpy_image_1.shape == (
        224,
        224,
        3,
    )

    assert numpy_image_1.dtype == np.float32
    assert float(numpy_image_1.min()) >= 0.0
    assert float(numpy_image_1.max()) <= 1.0

    assert np.array_equal(
        numpy_image_1,
        numpy_image_2,
    )

    tensor_1 = preprocessor.to_tensor(
        image_path
    )

    tensor_2 = preprocessor.to_tensor(
        image_path
    )

    assert tensor_1.shape == (
        3,
        224,
        224,
    )

    assert tensor_1.dtype == torch.float32
    assert torch.isfinite(tensor_1).all()
    assert torch.equal(tensor_1, tensor_2)


def test_dataset_split_lengths_and_metadata(
    preprocessor: ImagePreprocessor,
) -> None:
    train_dataset = build_development_dataset(
        manifest_path=MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        preprocessor=preprocessor,
        split="train",
        output_mode="numpy",
    )

    validation_dataset = build_development_dataset(
        manifest_path=MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        preprocessor=preprocessor,
        split="validation",
        output_mode="pil",
    )

    test_dataset = build_evaluation_dataset(
        manifest_path=MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        preprocessor=preprocessor,
        split="test",
        output_mode="tensor",
    )

    assert len(train_dataset) == 700
    assert len(validation_dataset) == 150
    assert len(test_dataset) == 150

    train_sample = train_dataset[0]

    assert train_sample["split"] == "train"
    assert train_sample["class_name"] in {
        "real",
        "ai_generated",
    }

    assert train_sample["label"] in {0, 1}

    assert isinstance(
        train_sample["image"],
        np.ndarray,
    )

    validation_sample = validation_dataset[0]

    assert validation_sample["split"] == "validation"

    assert isinstance(
        validation_sample["image"],
        Image.Image,
    )

    test_sample = test_dataset[0]

    assert test_sample["split"] == "test"

    assert isinstance(
        test_sample["image"],
        torch.Tensor,
    )


def test_tensor_dataloader_batch_contract(
    preprocessor: ImagePreprocessor,
) -> None:
    train_dataset = build_development_dataset(
        manifest_path=MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        preprocessor=preprocessor,
        split="train",
        output_mode="tensor",
    )

    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (
        8,
        3,
        224,
        224,
    )

    assert batch["image"].dtype == torch.float32

    assert batch["label"].shape == (8,)
    assert batch["label"].dtype == torch.int64

    assert len(batch["image_id"]) == 8
    assert len(batch["path"]) == 8
    assert torch.isfinite(batch["image"]).all()


def test_preprocessing_does_not_modify_raw_files(
    preprocessor: ImagePreprocessor,
) -> None:
    records = load_manifest(MANIFEST_PATH)

    sample_records = records[::125]

    for record in sample_records:
        image_path = record.absolute_path(
            PROJECT_ROOT
        )

        before_hash = sha256_file(image_path)

        preprocessor.load(image_path)
        preprocessor.to_numpy(image_path)
        preprocessor.to_tensor(image_path)

        after_hash = sha256_file(image_path)

        assert before_hash == record.copied_sha256
        assert after_hash == record.copied_sha256
