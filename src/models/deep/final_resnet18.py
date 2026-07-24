from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from src.models.classical.linear_svm import (
    calculate_binary_metrics,
)
from src.models.deep.resnet18 import (
    build_frozen_resnet18,
    train_one_epoch,
)


@dataclass(slots=True)
class FinalResNet18Fit:
    model: nn.Module
    record_count: int
    selected_epochs: int
    train_losses: list[float]
    total_parameters: int
    trainable_parameters: int


@dataclass(frozen=True, slots=True)
class FinalResNet18Evaluation:
    image_ids: tuple[str, ...]
    labels: np.ndarray
    predictions: np.ndarray
    positive_probabilities: np.ndarray
    loss: float
    metrics: dict[str, Any]


def fit_final_resnet18(
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool,
    selected_epochs: int,
) -> FinalResNet18Fit:
    if selected_epochs <= 0:
        raise ValueError(
            "Selected epoch count must be positive"
        )

    record_count = len(loader.dataset)

    if record_count <= 0:
        raise ValueError(
            "Final development dataset is empty"
        )

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if trainable_names != [
        "fc.weight",
        "fc.bias",
    ]:
        raise RuntimeError(
            "Only the ResNet-18 head may be trainable"
        )

    train_losses: list[float] = []

    for _ in range(selected_epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
        )

        train_losses.append(
            float(train_loss)
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters != 1026:
        raise RuntimeError(
            "Unexpected final trainable parameter count"
        )

    return FinalResNet18Fit(
        model=model,
        record_count=record_count,
        selected_epochs=selected_epochs,
        train_losses=train_losses,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )


def evaluate_final_resnet18(
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool,
) -> FinalResNet18Evaluation:
    model.eval()

    total_loss = 0.0
    total_records = 0

    image_ids: list[str] = []
    labels_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    labels,
                )

            if logits.ndim != 2:
                raise ValueError(
                    "ResNet-18 logits must be two-dimensional"
                )

            if logits.shape != (
                labels.shape[0],
                2,
            ):
                raise ValueError(
                    "Unexpected ResNet-18 output shape"
                )

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            batch_size = int(
                labels.shape[0]
            )

            batch_ids = [
                str(value)
                for value in batch["image_id"]
            ]

            if len(batch_ids) != batch_size:
                raise ValueError(
                    "Image-ID batch length mismatch"
                )

            image_ids.extend(batch_ids)

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_records += batch_size

            labels_parts.append(
                labels.cpu().numpy().astype(
                    np.int64,
                    copy=False,
                )
            )

            prediction_parts.append(
                predictions.cpu().numpy().astype(
                    np.int64,
                    copy=False,
                )
            )

            probability_parts.append(
                probabilities[
                    :,
                    1,
                ].cpu().numpy().astype(
                    np.float64,
                    copy=False,
                )
            )

    if total_records == 0:
        raise RuntimeError(
            "Test loader produced no records"
        )

    if len(image_ids) != total_records:
        raise RuntimeError(
            "Final image-ID count mismatch"
        )

    if len(set(image_ids)) != total_records:
        raise RuntimeError(
            "Duplicate image IDs in final evaluation"
        )

    labels_array = np.concatenate(
        labels_parts
    )

    predictions_array = np.concatenate(
        prediction_parts
    )

    positive_probabilities = np.concatenate(
        probability_parts
    )

    metrics = calculate_binary_metrics(
        labels_array,
        predictions_array,
        positive_probabilities,
    )

    return FinalResNet18Evaluation(
        image_ids=tuple(image_ids),
        labels=labels_array,
        predictions=predictions_array,
        positive_probabilities=(
            positive_probabilities
        ),
        loss=total_loss / total_records,
        metrics=metrics,
    )


def build_resnet18_checkpoint(
    *,
    model: nn.Module,
    selected_epoch: int,
    random_seed: int,
    selection_report_sha256: str,
) -> dict[str, Any]:
    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if trainable_names != [
        "fc.weight",
        "fc.bias",
    ]:
        raise RuntimeError(
            "Checkpoint model violates frozen-head contract"
        )

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor
        in model.state_dict().items()
    }

    return {
        "schema_version": 1,
        "architecture": "resnet18",
        "weights": "IMAGENET1K_V1",
        "output_classes": 2,
        "backbone_frozen": True,
        "selected_epoch": int(selected_epoch),
        "random_seed": int(random_seed),
        "selection_report_sha256": (
            selection_report_sha256
        ),
        "state_dict": state_dict,
    }


def load_resnet18_checkpoint(
    path: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "ResNet-18 checkpoint must be a mapping"
        )

    required = {
        "schema_version",
        "architecture",
        "weights",
        "output_classes",
        "backbone_frozen",
        "selected_epoch",
        "random_seed",
        "selection_report_sha256",
        "state_dict",
    }

    if not required.issubset(checkpoint):
        missing = sorted(
            required - set(checkpoint)
        )

        raise ValueError(
            "Missing checkpoint fields: "
            + ", ".join(missing)
        )

    if checkpoint["schema_version"] != 1:
        raise ValueError(
            "Unsupported checkpoint schema"
        )

    if checkpoint["architecture"] != "resnet18":
        raise ValueError(
            "Unexpected checkpoint architecture"
        )

    if checkpoint["output_classes"] != 2:
        raise ValueError(
            "Unexpected checkpoint class count"
        )

    if checkpoint["backbone_frozen"] is not True:
        raise ValueError(
            "Checkpoint backbone must be frozen"
        )

    model = build_frozen_resnet18(
        weights_name=None,
        output_classes=2,
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()

    return model, checkpoint
