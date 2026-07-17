from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data import load_manifest
from src.features import sha256_file
from src.models.deep import (
    ManifestImageDataset,
    build_frozen_resnet18,
    build_resnet18_transforms,
    choose_best_epoch,
    evaluate_binary_classifier,
    seed_everything,
    train_one_epoch,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_resnet18.yaml"
)


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
        )

    return data


def resolve_project_path(
    relative_path: str,
) -> Path:
    root = PROJECT_ROOT.resolve()
    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
        raise ValueError(
            f"Path escapes project root: "
            f"{relative_path}"
        )

    return path


def temporary_path(
    path: Path,
) -> Path:
    return path.with_name(
        path.name + ".temporary"
    )


def write_json_atomic(
    *,
    path: Path,
    data: dict[str, Any],
) -> None:
    temporary = temporary_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if temporary.exists():
        temporary.unlink()

    try:
        temporary.write_text(
            json.dumps(
                data,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def write_history_atomic(
    *,
    path: Path,
    history: list[dict[str, Any]],
) -> None:
    temporary = temporary_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if temporary.exists():
        temporary.unlink()

    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                lineterminator="\n",
            )

            writer.writeheader()

            for row in history:
                metrics = row[
                    "validation_metrics"
                ]

                confusion = metrics[
                    "confusion_matrix"
                ]

                writer.writerow(
                    {
                        "epoch": row["epoch"],
                        "train_loss": format(
                            row["train_loss"],
                            ".17g",
                        ),
                        "validation_loss": format(
                            row[
                                "validation_loss"
                            ],
                            ".17g",
                        ),
                        "accuracy": format(
                            metrics["accuracy"],
                            ".17g",
                        ),
                        "balanced_accuracy": format(
                            metrics[
                                "balanced_accuracy"
                            ],
                            ".17g",
                        ),
                        "precision": format(
                            metrics["precision"],
                            ".17g",
                        ),
                        "recall": format(
                            metrics["recall"],
                            ".17g",
                        ),
                        "f1": format(
                            metrics["f1"],
                            ".17g",
                        ),
                        "roc_auc": format(
                            metrics["roc_auc"],
                            ".17g",
                        ),
                        "true_negative": (
                            confusion[0][0]
                        ),
                        "false_positive": (
                            confusion[0][1]
                        ),
                        "false_negative": (
                            confusion[1][0]
                        ),
                        "true_positive": (
                            confusion[1][1]
                        ),
                    }
                )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-backbone prototype "
            "ResNet-18 and select an epoch using "
            "the validation split only."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace existing validation-selection "
            "artifacts."
        ),
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config_file = load_yaml(config_path)

    config = config_file[
        "resnet18_selection"
    ]

    seed = int(config["random_seed"])

    training_split = str(
        config["splits"]["training"]
    )

    validation_split = str(
        config["splits"]["validation"]
    )

    if training_split != "train":
        raise ValueError(
            "Training split must be train"
        )

    if validation_split != "validation":
        raise ValueError(
            "Selection split must be validation"
        )

    if "test" in {
        training_split,
        validation_split,
    }:
        raise ValueError(
            "Test split cannot be used "
            "during model selection"
        )

    manifest_path = resolve_project_path(
        config["manifest"]
    )

    report_path = resolve_project_path(
        config["outputs"][
            "selection_report"
        ]
    )

    history_path = resolve_project_path(
        config["outputs"][
            "training_history"
        ]
    )

    existing = [
        path
        for path in [
            report_path,
            history_path,
        ]
        if path.exists()
    ]

    if existing and not args.force:
        raise FileExistsError(
            "ResNet-18 selection artifacts "
            "already exist. Use --force to "
            "replace them: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    model_config = config["model"]
    preprocessing = config[
        "preprocessing"
    ]
    training = config["training"]

    if (
        model_config["architecture"]
        != "resnet18"
    ):
        raise ValueError(
            "Architecture must be resnet18"
        )

    if not bool(
        model_config["freeze_backbone"]
    ):
        raise ValueError(
            "Prototype ResNet-18 backbone "
            "must remain frozen"
        )

    if training["optimizer"] != "adamw":
        raise ValueError(
            "Prototype optimizer must be AdamW"
        )

    if int(training["epochs"]) != 2:
        raise ValueError(
            "Prototype selection requires "
            "exactly two epochs"
        )

    if int(training["num_workers"]) != 0:
        raise ValueError(
            "Prototype selection uses "
            "num_workers=0 for reproducibility"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for prototype "
            "ResNet-18 training"
        )

    seed_everything(
        seed,
        deterministic=bool(
            training["deterministic"]
        ),
    )

    records = load_manifest(
        manifest_path
    )

    training_records = [
        record
        for record in records
        if record.split == training_split
    ]

    validation_records = [
        record
        for record in records
        if record.split == validation_split
    ]

    if len(training_records) != 700:
        raise ValueError(
            "Expected 700 training records"
        )

    if len(validation_records) != 150:
        raise ValueError(
            "Expected 150 validation records"
        )

    training_ids = {
        record.image_id
        for record in training_records
    }

    validation_ids = {
        record.image_id
        for record in validation_records
    }

    overlap = (
        training_ids
        & validation_ids
    )

    if overlap:
        raise ValueError(
            "Training and validation IDs overlap"
        )

    training_counts = Counter(
        int(record.label_id)
        for record in training_records
    )

    validation_counts = Counter(
        int(record.label_id)
        for record in validation_records
    )

    if training_counts != Counter(
        {
            0: 350,
            1: 350,
        }
    ):
        raise ValueError(
            "Unexpected training class counts"
        )

    if validation_counts != Counter(
        {
            0: 75,
            1: 75,
        }
    ):
        raise ValueError(
            "Unexpected validation class counts"
        )

    (
        training_transform,
        validation_transform,
    ) = build_resnet18_transforms(
        crop_size=int(
            preprocessing["crop_size"]
        ),
        resize_size=int(
            preprocessing["resize_size"]
        ),
        mean=preprocessing["mean"],
        standard_deviation=(
            preprocessing[
                "standard_deviation"
            ]
        ),
    )

    training_dataset = (
        ManifestImageDataset(
            records=training_records,
            project_root=PROJECT_ROOT,
            transform=training_transform,
        )
    )

    validation_dataset = (
        ManifestImageDataset(
            records=validation_records,
            project_root=PROJECT_ROOT,
            transform=validation_transform,
        )
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    batch_size = int(
        training["batch_size"]
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = build_frozen_resnet18(
        weights_name=str(
            model_config["weights"]
        ),
        output_classes=int(
            model_config["output_classes"]
        ),
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

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if trainable_parameters != 1026:
        raise RuntimeError(
            "Unexpected trainable parameter count"
        )

    device = torch.device("cuda:0")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=float(
            training["learning_rate"]
        ),
        weight_decay=float(
            training["weight_decay"]
        ),
    )

    mixed_precision = bool(
        training["mixed_precision"]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision,
    )

    history: list[
        dict[str, Any]
    ] = []

    print(
        "========== RESNET-18 VALIDATION TRAINING =========="
    )
    print(
        "Training records:",
        len(training_dataset),
    )
    print(
        "Validation records:",
        len(validation_dataset),
    )
    print("Test images opened: NO")
    print("Device:", torch.cuda.get_device_name(0))
    print(
        "Trainable parameters:",
        trainable_parameters,
    )
    print("Epochs:", training["epochs"])
    print("Batch size:", batch_size)
    print()

    for epoch in range(
        1,
        int(training["epochs"]) + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=(
                mixed_precision
            ),
        )

        validation_result = (
            evaluate_binary_classifier(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                mixed_precision=(
                    mixed_precision
                ),
            )
        )

        metrics = validation_result[
            "metrics"
        ]

        row = {
            "epoch": epoch,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": float(
                validation_result["loss"]
            ),
            "validation_metrics": metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{training['epochs']} "
            f"| train loss: {train_loss:.4f} "
            f"| validation loss: "
            f"{validation_result['loss']:.4f} "
            f"| F1: {metrics['f1']:.4f} "
            f"| ROC-AUC: "
            f"{metrics['roc_auc']:.4f} "
            f"| accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

    selected = choose_best_epoch(
        history
    )

    report = {
        "schema_version": 1,
        "model": "pretrained_resnet18",
        "architecture": "resnet18",
        "weights": (
            model_config["weights"]
        ),
        "backbone_frozen": True,
        "random_seed": seed,
        "selection_config": (
            config_path.relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        ),
        "selection_config_sha256": (
            sha256_file(config_path)
        ),
        "manifest": (
            manifest_path.relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        ),
        "manifest_sha256": (
            sha256_file(manifest_path)
        ),
        "class_mapping": {
            str(key): value
            for key, value in config[
                "class_mapping"
            ].items()
        },
        "development_data": {
            "training_split": (
                training_split
            ),
            "training_records": (
                len(training_records)
            ),
            "training_class_counts": {
                str(label): count
                for label, count in sorted(
                    training_counts.items()
                )
            },
            "validation_split": (
                validation_split
            ),
            "validation_records": (
                len(validation_records)
            ),
            "validation_class_counts": {
                str(label): count
                for label, count in sorted(
                    validation_counts.items()
                )
            },
            "image_id_overlap": len(
                overlap
            ),
            "test_images_opened": False,
        },
        "model_contract": {
            "total_parameters": (
                total_parameters
            ),
            "trainable_parameters": (
                trainable_parameters
            ),
            "trainable_tensors": (
                trainable_names
            ),
            "output_classes": int(
                model_config[
                    "output_classes"
                ]
            ),
        },
        "preprocessing": {
            "training": [
                "RandomResizedCrop(224)",
                "RandomHorizontalFlip(p=0.5)",
                "ToTensor",
                "ImageNet normalization",
            ],
            "validation": [
                "Resize(256)",
                "CenterCrop(224)",
                "ToTensor",
                "ImageNet normalization",
            ],
            "mean": preprocessing["mean"],
            "standard_deviation": (
                preprocessing[
                    "standard_deviation"
                ]
            ),
        },
        "training_contract": {
            "epochs": int(
                training["epochs"]
            ),
            "batch_size": batch_size,
            "learning_rate": float(
                training["learning_rate"]
            ),
            "weight_decay": float(
                training["weight_decay"]
            ),
            "optimizer": (
                training["optimizer"]
            ),
            "num_workers": int(
                training["num_workers"]
            ),
            "mixed_precision": (
                mixed_precision
            ),
            "deterministic": bool(
                training["deterministic"]
            ),
        },
        "selection_rule": {
            "primary_metric": (
                config["selection"][
                    "primary_metric"
                ]
            ),
            "tie_breakers": list(
                config["selection"][
                    "tie_breakers"
                ]
            ),
        },
        "history": history,
        "selected_epoch": int(
            selected["epoch"]
        ),
        "selected_validation_loss": float(
            selected["validation_loss"]
        ),
        "selected_validation_metrics": (
            selected[
                "validation_metrics"
            ]
        ),
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "torchvision": (
                torchvision.__version__
            ),
            "cuda_runtime": (
                torch.version.cuda
            ),
        },
    }

    write_history_atomic(
        path=history_path,
        history=history,
    )

    report[
        "training_history_sha256"
    ] = sha256_file(history_path)

    write_json_atomic(
        path=report_path,
        data=report,
    )

    print()
    print(
        "Selected epoch:",
        report["selected_epoch"],
    )
    print(
        "Selected validation F1:",
        f"{report['selected_validation_metrics']['f1']:.4f}",
    )
    print(
        "Selected validation ROC-AUC:",
        f"{report['selected_validation_metrics']['roc_auc']:.4f}",
    )
    print(
        "Selection report:",
        report_path.relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Training history:",
        history_path.relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Selection report SHA-256:",
        sha256_file(report_path),
    )
    print()
    print(
        "PASS: ResNet-18 epoch selected "
        "using training and validation only."
    )
    print(
        "PASS: frozen backbone received "
        "no gradients."
    )
    print(
        "PASS: test images were not opened."
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
