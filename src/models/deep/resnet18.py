from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.transforms import (
    InterpolationMode,
)

from src.models.classical.linear_svm import (
    calculate_binary_metrics,
)


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        *,
        records: Sequence[Any],
        project_root: Path,
        transform: Any,
    ) -> None:
        if not records:
            raise ValueError(
                "ManifestImageDataset requires records"
            )

        self.records = tuple(records)
        self.project_root = project_root.resolve()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        record = self.records[index]

        image_path = record.absolute_path(
            self.project_root
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            tensor = self.transform(rgb_image)

        if tensor.shape != (3, 224, 224):
            raise ValueError(
                "Unexpected ResNet-18 tensor shape: "
                f"{tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"Non-finite tensor for {record.image_id}"
            )

        return {
            "image": tensor,
            "label": int(record.label_id),
            "image_id": str(record.image_id),
            "class_name": str(record.label),
            "split": str(record.split),
        }


def seed_everything(
    seed: int,
    *,
    deterministic: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = (
        deterministic
    )

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )


def build_resnet18_transforms(
    *,
    crop_size: int,
    resize_size: int,
    mean: Sequence[float],
    standard_deviation: Sequence[float],
) -> tuple[Any, Any]:
    if crop_size != 224:
        raise ValueError(
            "Prototype ResNet-18 crop size must be 224"
        )

    normalization = transforms.Normalize(
        mean=list(mean),
        std=list(standard_deviation),
    )

    training_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                crop_size,
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            ),
            transforms.CenterCrop(
                crop_size
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )

    return (
        training_transform,
        evaluation_transform,
    )


def resolve_resnet18_weights(
    weights_name: str | None,
) -> ResNet18_Weights | None:
    if weights_name is None:
        return None

    if weights_name == "IMAGENET1K_V1":
        return (
            ResNet18_Weights.IMAGENET1K_V1
        )

    raise ValueError(
        f"Unsupported ResNet-18 weights: "
        f"{weights_name}"
    )


def build_frozen_resnet18(
    *,
    weights_name: str | None,
    output_classes: int,
) -> nn.Module:
    if output_classes != 2:
        raise ValueError(
            "Prototype ResNet-18 requires "
            "two output classes"
        )

    model = resnet18(
        weights=resolve_resnet18_weights(
            weights_name
        )
    )

    input_features = int(
        model.fc.in_features
    )

    if input_features != 512:
        raise ValueError(
            "Unexpected ResNet-18 head dimension"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(
        input_features,
        output_classes,
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
            "Only the ResNet-18 classification "
            "head may be trainable"
        )

    return model


def set_head_training_mode(
    model: nn.Module,
) -> None:
    model.eval()
    model.fc.train()

    if model.training:
        raise RuntimeError(
            "Frozen ResNet-18 backbone "
            "must remain in evaluation mode"
        )

    if not model.fc.training:
        raise RuntimeError(
            "ResNet-18 classification head "
            "must be in training mode"
        )


def validate_gradient_contract(
    model: nn.Module,
) -> None:
    trainable_with_grad = [
        name
        for name, parameter
        in model.named_parameters()
        if (
            parameter.requires_grad
            and parameter.grad is not None
        )
    ]

    frozen_with_grad = [
        name
        for name, parameter
        in model.named_parameters()
        if (
            not parameter.requires_grad
            and parameter.grad is not None
        )
    ]

    if trainable_with_grad != [
        "fc.weight",
        "fc.bias",
    ]:
        raise RuntimeError(
            "Expected gradients only for "
            "fc.weight and fc.bias"
        )

    if frozen_with_grad:
        raise RuntimeError(
            "Frozen backbone received gradients: "
            + ", ".join(frozen_with_grad)
        )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    set_head_training_mode(model)

    total_loss = 0.0
    total_records = 0
    gradient_checked = False

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
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

        scaler.scale(loss).backward()

        if not gradient_checked:
            validate_gradient_contract(
                model
            )
            gradient_checked = True

        scaler.step(optimizer)
        scaler.update()

        batch_size = int(
            labels.shape[0]
        )

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )

        total_records += batch_size

    if total_records == 0:
        raise RuntimeError(
            "Training loader produced no records"
        )

    if not gradient_checked:
        raise RuntimeError(
            "Gradient contract was not checked"
        )

    return total_loss / total_records


def evaluate_binary_classifier(
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_records = 0

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
            "Evaluation loader produced no records"
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

    return {
        "loss": total_loss / total_records,
        "metrics": metrics,
        "labels": labels_array,
        "predictions": predictions_array,
        "positive_probabilities": (
            positive_probabilities
        ),
    }


def choose_best_epoch(
    history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not history:
        raise ValueError(
            "Training history cannot be empty"
        )

    return max(
        history,
        key=lambda row: (
            float(
                row[
                    "validation_metrics"
                ]["f1"]
            ),
            float(
                row[
                    "validation_metrics"
                ]["roc_auc"]
            ),
            -int(row["epoch"]),
        ),
    )
