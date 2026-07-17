from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.models.deep import (
    ManifestImageDataset,
    build_frozen_resnet18,
    build_resnet18_transforms,
    choose_best_epoch,
    set_head_training_mode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_resnet18.yaml"
)


@dataclass(frozen=True)
class FakeRecord:
    image_id: str
    label_id: int
    label: str
    split: str
    relative_path: str

    def absolute_path(
        self,
        project_root: Path,
    ) -> Path:
        return (
            project_root
            / self.relative_path
        ).resolve()


def load_config():
    return yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )["resnet18_selection"]


def test_resnet18_selection_configuration() -> None:
    config = load_config()

    assert config["splits"] == {
        "training": "train",
        "validation": "validation",
    }

    assert "test" not in set(
        config["splits"].values()
    )

    assert config["training"]["epochs"] == 2

    assert (
        config["training"]["num_workers"]
        == 0
    )

    assert (
        config["model"]["freeze_backbone"]
        is True
    )


def test_frozen_resnet18_contract() -> None:
    model = build_frozen_resnet18(
        weights_name=None,
        output_classes=2,
    )

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_names == [
        "fc.weight",
        "fc.bias",
    ]

    assert trainable_parameters == 1026

    set_head_training_mode(model)

    assert model.training is False
    assert model.fc.training is True
    assert model.bn1.training is False


def test_epoch_selection_rule() -> None:
    history = [
        {
            "epoch": 1,
            "validation_metrics": {
                "f1": 0.8,
                "roc_auc": 0.9,
            },
        },
        {
            "epoch": 2,
            "validation_metrics": {
                "f1": 0.8,
                "roc_auc": 0.9,
            },
        },
    ]

    selected = choose_best_epoch(
        history
    )

    assert selected["epoch"] == 1

    history[1]["validation_metrics"][
        "roc_auc"
    ] = 0.91

    selected = choose_best_epoch(
        history
    )

    assert selected["epoch"] == 2


def test_manifest_dataset_and_transforms(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"

    image_array = np.full(
        (40, 50, 3),
        128,
        dtype=np.uint8,
    )

    Image.fromarray(
        image_array,
        mode="RGB",
    ).save(image_path)

    config = load_config()
    preprocessing = config[
        "preprocessing"
    ]

    _, evaluation_transform = (
        build_resnet18_transforms(
            crop_size=preprocessing[
                "crop_size"
            ],
            resize_size=preprocessing[
                "resize_size"
            ],
            mean=preprocessing["mean"],
            standard_deviation=(
                preprocessing[
                    "standard_deviation"
                ]
            ),
        )
    )

    record = FakeRecord(
        image_id="sample-1",
        label_id=1,
        label="ai_generated",
        split="validation",
        relative_path="sample.png",
    )

    dataset = ManifestImageDataset(
        records=[record],
        project_root=tmp_path,
        transform=evaluation_transform,
    )

    sample = dataset[0]

    assert sample["image"].shape == (
        3,
        224,
        224,
    )

    assert sample["image"].dtype == (
        torch.float32
    )

    assert torch.isfinite(
        sample["image"]
    ).all()

    assert sample["label"] == 1
    assert sample["image_id"] == "sample-1"
    assert (
        sample["class_name"]
        == "ai_generated"
    )
