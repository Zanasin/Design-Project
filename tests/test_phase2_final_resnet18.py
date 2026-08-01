from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.deep import (
    build_frozen_resnet18,
    build_resnet18_checkpoint,
    evaluate_final_resnet18,
    load_resnet18_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_resnet18_final.yaml"
)

SELECTION_REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "prototype"
    / "resnet18"
    / "validation_selection.json"
)


class TinyEvaluationDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(
        self,
        index: int,
    ):
        return {
            "image": torch.full(
                (3, 4, 4),
                float(index),
                dtype=torch.float32,
            ),
            "label": index % 2,
            "image_id": f"image-{index}",
        }


class TinyClassifier(nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        mean = inputs.mean(
            dim=(1, 2, 3)
        )

        return torch.stack(
            [
                -mean,
                mean,
            ],
            dim=1,
        )


def test_final_configuration_contract() -> None:
    data = yaml.safe_load(
        FINAL_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    final = data["final_resnet18"]

    assert final[
        "development_splits"
    ] == [
        "train",
        "validation",
    ]

    assert final["test_split"] == "test"

    assert "test" not in final[
        "development_splits"
    ]

    paths = list(
        final["artifacts"].values()
    )

    assert len(paths) == len(set(paths))


def test_locked_epoch_matches_selection() -> None:
    report = json.loads(
        SELECTION_REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert report["selected_epoch"] == 2
    assert report["backbone_frozen"] is True
    assert (
        report["development_data"][
            "test_images_opened"
        ]
        is False
    )


def test_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    model = build_frozen_resnet18(
        weights_name=None,
        output_classes=2,
    )

    checkpoint = build_resnet18_checkpoint(
        model=model,
        selected_epoch=2,
        random_seed=42,
        selection_report_sha256="a" * 64,
    )

    path = tmp_path / "model.pt"

    torch.save(
        checkpoint,
        path,
    )

    loaded_model, loaded = (
        load_resnet18_checkpoint(path)
    )

    assert loaded["selected_epoch"] == 2
    assert loaded["random_seed"] == 42

    for name, tensor in (
        model.state_dict().items()
    ):
        assert torch.equal(
            tensor,
            loaded_model.state_dict()[name],
        )


def test_final_evaluation_collects_ids() -> None:
    loader = DataLoader(
        TinyEvaluationDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    result = evaluate_final_resnet18(
        model=TinyClassifier(),
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )

    assert result.image_ids == (
        "image-0",
        "image-1",
        "image-2",
        "image-3",
    )

    assert result.labels.shape == (4,)
    assert result.predictions.shape == (4,)

    assert (
        result.positive_probabilities.shape
        == (4,)
    )

    assert np.isfinite(
        result.positive_probabilities
    ).all()

    assert result.metrics[
        "confusion_matrix"
    ][0][0] >= 0
