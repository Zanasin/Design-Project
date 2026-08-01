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
from src.models.classical.linear_svm import (
    calculate_binary_metrics,
)
from src.models.deep import (
    ManifestImageDataset,
    build_frozen_resnet18,
    build_resnet18_checkpoint,
    build_resnet18_transforms,
    evaluate_final_resnet18,
    fit_final_resnet18,
    load_resnet18_checkpoint,
    seed_everything,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_resnet18_final.yaml"
)

CLASS_NAMES = {
    0: "real",
    1: "ai_generated",
}


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}"
        )

    return data


def load_json(
    path: Path,
) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {path}"
        )

    return data


def resolve_project_path(
    relative_path: str,
) -> Path:
    root = PROJECT_ROOT.resolve()
    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
        raise ValueError(
            f"Path escapes project root: {relative_path}"
        )

    return path


def temporary_path(
    path: Path,
) -> Path:
    return path.with_name(
        path.name + ".temporary"
    )


def write_json(
    *,
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_predictions(
    *,
    path: Path,
    image_ids: tuple[str, ...],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "positive_probability",
                "correct",
            ],
            lineterminator="\n",
        )

        writer.writeheader()

        for (
            image_id,
            true_label,
            predicted_label,
            probability,
        ) in zip(
            image_ids,
            labels,
            predictions,
            probabilities,
            strict=True,
        ):
            true_value = int(true_label)
            predicted_value = int(
                predicted_label
            )

            writer.writerow(
                {
                    "image_id": image_id,
                    "true_label": true_value,
                    "true_class": (
                        CLASS_NAMES[true_value]
                    ),
                    "predicted_label": (
                        predicted_value
                    ),
                    "predicted_class": (
                        CLASS_NAMES[
                            predicted_value
                        ]
                    ),
                    "positive_probability": (
                        format(
                            float(probability),
                            ".17g",
                        )
                    ),
                    "correct": int(
                        true_value
                        == predicted_value
                    ),
                }
            )


def validate_locked_selection(
    *,
    selection_path: Path,
    selection: dict[str, Any],
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    int,
]:
    if (
        selection["model"]
        != "pretrained_resnet18"
    ):
        raise ValueError(
            "Locked selection is not ResNet-18"
        )

    if selection["architecture"] != "resnet18":
        raise ValueError(
            "Unexpected locked architecture"
        )

    if selection["backbone_frozen"] is not True:
        raise ValueError(
            "Locked backbone is not frozen"
        )

    if (
        selection["development_data"][
            "test_images_opened"
        ]
        is not False
    ):
        raise ValueError(
            "Selection report indicates test access"
        )

    selection_config_path = (
        resolve_project_path(
            selection["selection_config"]
        )
    )

    manifest_path = resolve_project_path(
        selection["manifest"]
    )

    if (
        sha256_file(selection_config_path)
        != selection[
            "selection_config_sha256"
        ]
    ):
        raise ValueError(
            "ResNet-18 selection config changed"
        )

    if (
        sha256_file(manifest_path)
        != selection["manifest_sha256"]
    ):
        raise ValueError(
            "Prototype manifest changed"
        )

    selection_config_file = load_yaml(
        selection_config_path
    )

    config = selection_config_file[
        "resnet18_selection"
    ]

    history_path = resolve_project_path(
        config["outputs"][
            "training_history"
        ]
    )

    if (
        sha256_file(history_path)
        != selection[
            "training_history_sha256"
        ]
    ):
        raise ValueError(
            "ResNet-18 training history changed"
        )

    selected_epoch = int(
        selection["selected_epoch"]
    )

    configured_epochs = int(
        config["training"]["epochs"]
    )

    if selected_epoch != 2:
        raise ValueError(
            "Prototype selected epoch must be 2"
        )

    if selected_epoch > configured_epochs:
        raise ValueError(
            "Selected epoch exceeds selection run"
        )

    if (
        selection["selection_config_sha256"]
        != sha256_file(selection_config_path)
    ):
        raise ValueError(
            "Selection configuration hash mismatch"
        )

    if not selection_path.is_file():
        raise FileNotFoundError(
            "Selection report is missing"
        )

    return (
        config,
        selection_config_path,
        manifest_path,
        selected_epoch,
    )


def verify_final_artifacts(
    *,
    config_path: Path,
    config_file: dict[str, Any],
) -> dict[str, Any]:
    final = config_file[
        "final_resnet18"
    ]

    selection_path = resolve_project_path(
        final["selection_report"]
    )

    model_path = resolve_project_path(
        final["artifacts"]["model"]
    )

    report_path = resolve_project_path(
        final["artifacts"][
            "test_evaluation"
        ]
    )

    predictions_path = resolve_project_path(
        final["artifacts"][
            "test_predictions"
        ]
    )

    for path in [
        selection_path,
        model_path,
        report_path,
        predictions_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

    report = load_json(report_path)
    selection = load_json(selection_path)

    manifest_path = resolve_project_path(
        report["manifest"]
    )

    checks = [
        (
            report["final_config_sha256"],
            sha256_file(config_path),
            "final configuration",
        ),
        (
            report["selection_report_sha256"],
            sha256_file(selection_path),
            "selection report",
        ),
        (
            report["manifest_sha256"],
            sha256_file(manifest_path),
            "manifest",
        ),
        (
            report["artifacts"][
                "model"
            ]["sha256"],
            sha256_file(model_path),
            "model",
        ),
        (
            report["artifacts"][
                "test_predictions"
            ]["sha256"],
            sha256_file(predictions_path),
            "predictions",
        ),
    ]

    for recorded, actual, name in checks:
        if recorded != actual:
            raise ValueError(
                f"{name} SHA-256 mismatch"
            )

    model, checkpoint = (
        load_resnet18_checkpoint(
            model_path
        )
    )

    if (
        checkpoint["selected_epoch"]
        != report["selected_epoch"]
    ):
        raise ValueError(
            "Checkpoint epoch mismatch"
        )

    if (
        checkpoint[
            "selection_report_sha256"
        ]
        != report[
            "selection_report_sha256"
        ]
    ):
        raise ValueError(
            "Checkpoint selection hash mismatch"
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters != 1026:
        raise ValueError(
            "Loaded checkpoint violates head contract"
        )

    with predictions_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    expected_records = int(
        report["test_data"]["records"]
    )

    if len(rows) != expected_records:
        raise ValueError(
            "Prediction row count mismatch"
        )

    prediction_ids = [
        row["image_id"]
        for row in rows
    ]

    if len(set(prediction_ids)) != expected_records:
        raise ValueError(
            "Prediction image IDs are not unique"
        )

    manifest_records = load_manifest(
        manifest_path
    )

    expected_test_ids = {
        record.image_id
        for record in manifest_records
        if record.split == "test"
    }

    if set(prediction_ids) != expected_test_ids:
        raise ValueError(
            "Prediction IDs do not match test manifest"
        )

    labels = np.asarray(
        [
            int(row["true_label"])
            for row in rows
        ],
        dtype=np.int64,
    )

    predictions = np.asarray(
        [
            int(row["predicted_label"])
            for row in rows
        ],
        dtype=np.int64,
    )

    probabilities = np.asarray(
        [
            float(
                row[
                    "positive_probability"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    recalculated = calculate_binary_metrics(
        labels,
        predictions,
        probabilities,
    )

    recorded_metrics = report[
        "test_metrics"
    ]

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        if not np.isclose(
            recalculated[metric],
            recorded_metrics[metric],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Recorded {metric} does not "
                "match predictions"
            )

    if (
        recalculated["confusion_matrix"]
        != recorded_metrics[
            "confusion_matrix"
        ]
    ):
        raise ValueError(
            "Confusion matrix mismatch"
        )

    if (
        selection["selected_epoch"]
        != report["selected_epoch"]
    ):
        raise ValueError(
            "Selection and final epoch mismatch"
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the locked frozen-backbone "
            "ResNet-18 on train plus validation and "
            "evaluate the test split once."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify existing artifacts without "
            "training or opening test images."
        ),
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config_file = load_yaml(config_path)
    final = config_file[
        "final_resnet18"
    ]

    development_splits = list(
        final["development_splits"]
    )

    test_split = str(
        final["test_split"]
    )

    if development_splits != [
        "train",
        "validation",
    ]:
        raise ValueError(
            "Development splits must be "
            "train and validation"
        )

    if test_split != "test":
        raise ValueError(
            "Final evaluation split must be test"
        )

    if test_split in development_splits:
        raise ValueError(
            "Test split cannot be used for fitting"
        )

    if args.verify_only:
        report = verify_final_artifacts(
            config_path=config_path,
            config_file=config_file,
        )

        print(
            "Final ResNet-18 artifact "
            "verification passed."
        )
        print(
            "Selected epoch:",
            report["selected_epoch"],
        )
        print(
            "Development records:",
            report["fit_data"]["records"],
        )
        print(
            "Test records:",
            report["test_data"]["records"],
        )
        print(
            "Test F1:",
            f"{report['test_metrics']['f1']:.4f}",
        )
        print(
            "Test ROC-AUC:",
            f"{report['test_metrics']['roc_auc']:.4f}",
        )

        return 0

    selection_path = resolve_project_path(
        final["selection_report"]
    )

    model_path = resolve_project_path(
        final["artifacts"]["model"]
    )

    report_path = resolve_project_path(
        final["artifacts"][
            "test_evaluation"
        ]
    )

    predictions_path = resolve_project_path(
        final["artifacts"][
            "test_predictions"
        ]
    )

    final_paths = [
        model_path,
        report_path,
        predictions_path,
    ]

    existing = [
        path
        for path in final_paths
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Final ResNet-18 artifacts already "
            "exist. Use --verify-only instead: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for final ResNet-18"
        )

    selection = load_json(
        selection_path
    )

    (
        selection_config,
        selection_config_path,
        manifest_path,
        selected_epoch,
    ) = validate_locked_selection(
        selection_path=selection_path,
        selection=selection,
    )

    seed = int(
        selection_config["random_seed"]
    )

    training_config = selection_config[
        "training"
    ]

    model_config = selection_config[
        "model"
    ]

    preprocessing = selection_config[
        "preprocessing"
    ]

    seed_everything(
        seed,
        deterministic=bool(
            training_config["deterministic"]
        ),
    )

    records = load_manifest(
        manifest_path
    )

    split_records = {
        split: [
            record
            for record in records
            if record.split == split
        ]
        for split in [
            *development_splits,
            test_split,
        ]
    }

    expected_sizes = {
        "train": 700,
        "validation": 150,
        "test": 150,
    }

    for split, expected_size in (
        expected_sizes.items()
    ):
        if (
            len(split_records[split])
            != expected_size
        ):
            raise ValueError(
                f"Unexpected {split} record count"
            )

    id_sets = {
        split: {
            record.image_id
            for record
            in split_records[split]
        }
        for split in split_records
    }

    overlaps = {
        "train_validation": len(
            id_sets["train"]
            & id_sets["validation"]
        ),
        "train_test": len(
            id_sets["train"]
            & id_sets["test"]
        ),
        "validation_test": len(
            id_sets["validation"]
            & id_sets["test"]
        ),
    }

    if any(overlaps.values()):
        raise ValueError(
            f"Split image-ID overlap: {overlaps}"
        )

    development_records = [
        *split_records["train"],
        *split_records["validation"],
    ]

    development_counts = Counter(
        int(record.label_id)
        for record in development_records
    )

    test_counts = Counter(
        int(record.label_id)
        for record
        in split_records["test"]
    )

    if development_counts != Counter(
        {
            0: 425,
            1: 425,
        }
    ):
        raise ValueError(
            "Unexpected development class counts"
        )

    if test_counts != Counter(
        {
            0: 75,
            1: 75,
        }
    ):
        raise ValueError(
            "Unexpected test class counts"
        )

    (
        training_transform,
        evaluation_transform,
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

    development_dataset = (
        ManifestImageDataset(
            records=development_records,
            project_root=PROJECT_ROOT,
            transform=training_transform,
        )
    )

    test_dataset = ManifestImageDataset(
        records=split_records["test"],
        project_root=PROJECT_ROOT,
        transform=evaluation_transform,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    batch_size = int(
        training_config["batch_size"]
    )

    development_loader = DataLoader(
        development_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    test_loader = DataLoader(
        test_dataset,
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
            training_config["learning_rate"]
        ),
        weight_decay=float(
            training_config["weight_decay"]
        ),
    )

    mixed_precision = bool(
        training_config[
            "mixed_precision"
        ]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision,
    )

    fit = fit_final_resnet18(
        model=model,
        loader=development_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        mixed_precision=mixed_precision,
        selected_epochs=selected_epoch,
    )

    evaluation = evaluate_final_resnet18(
        model=fit.model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        mixed_precision=mixed_precision,
    )

    model_temporary = temporary_path(
        model_path
    )

    report_temporary = temporary_path(
        report_path
    )

    predictions_temporary = temporary_path(
        predictions_path
    )

    temporary_paths = [
        model_temporary,
        report_temporary,
        predictions_temporary,
    ]

    for path in temporary_paths:
        if path.exists():
            path.unlink()

    try:
        model_temporary.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = build_resnet18_checkpoint(
            model=fit.model,
            selected_epoch=selected_epoch,
            random_seed=seed,
            selection_report_sha256=(
                sha256_file(selection_path)
            ),
        )

        torch.save(
            checkpoint,
            model_temporary,
        )

        write_predictions(
            path=predictions_temporary,
            image_ids=evaluation.image_ids,
            labels=evaluation.labels,
            predictions=evaluation.predictions,
            probabilities=(
                evaluation.positive_probabilities
            ),
        )

        report = {
            "schema_version": 1,
            "model": "pretrained_resnet18",
            "architecture": "resnet18",
            "weights": model_config["weights"],
            "backbone_frozen": True,
            "selected_epoch": selected_epoch,
            "selected_validation_metrics": (
                selection[
                    "selected_validation_metrics"
                ]
            ),
            "final_config": (
                config_path.relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix()
            ),
            "final_config_sha256": (
                sha256_file(config_path)
            ),
            "selection_report": (
                selection_path.relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix()
            ),
            "selection_report_sha256": (
                sha256_file(selection_path)
            ),
            "selection_config": (
                selection_config_path.relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix()
            ),
            "selection_config_sha256": (
                sha256_file(
                    selection_config_path
                )
            ),
            "manifest": (
                manifest_path.relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix()
            ),
            "manifest_sha256": (
                sha256_file(manifest_path)
            ),
            "random_seed": seed,
            "model_contract": {
                "total_parameters": (
                    fit.total_parameters
                ),
                "trainable_parameters": (
                    fit.trainable_parameters
                ),
                "trainable_tensors": [
                    "fc.weight",
                    "fc.bias",
                ],
                "output_classes": 2,
            },
            "training_contract": {
                "optimizer": (
                    training_config["optimizer"]
                ),
                "learning_rate": float(
                    training_config[
                        "learning_rate"
                    ]
                ),
                "weight_decay": float(
                    training_config[
                        "weight_decay"
                    ]
                ),
                "batch_size": batch_size,
                "num_workers": 0,
                "mixed_precision": (
                    mixed_precision
                ),
                "deterministic": bool(
                    training_config[
                        "deterministic"
                    ]
                ),
            },
            "fit_data": {
                "splits": development_splits,
                "records": fit.record_count,
                "class_counts": {
                    str(label): count
                    for label, count in sorted(
                        development_counts.items()
                    )
                },
                "split_image_id_overlaps": (
                    overlaps
                ),
            },
            "fit_result": {
                "epochs": fit.selected_epochs,
                "train_losses": (
                    fit.train_losses
                ),
            },
            "test_data": {
                "split": test_split,
                "records": len(test_dataset),
                "class_counts": {
                    str(label): count
                    for label, count in sorted(
                        test_counts.items()
                    )
                },
                "loss": evaluation.loss,
            },
            "test_metrics": evaluation.metrics,
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
            "artifacts": {
                "model": {
                    "path": (
                        model_path.relative_to(
                            PROJECT_ROOT.resolve()
                        ).as_posix()
                    ),
                    "bytes": (
                        model_temporary.stat().st_size
                    ),
                    "sha256": (
                        sha256_file(
                            model_temporary
                        )
                    ),
                },
                "test_predictions": {
                    "path": (
                        predictions_path.relative_to(
                            PROJECT_ROOT.resolve()
                        ).as_posix()
                    ),
                    "bytes": (
                        predictions_temporary.stat().st_size
                    ),
                    "sha256": (
                        sha256_file(
                            predictions_temporary
                        )
                    ),
                },
            },
        }

        write_json(
            path=report_temporary,
            data=report,
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_temporary.replace(
            model_path
        )

        predictions_temporary.replace(
            predictions_path
        )

        report_temporary.replace(
            report_path
        )

    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()

    verified = verify_final_artifacts(
        config_path=config_path,
        config_file=config_file,
    )

    metrics = verified["test_metrics"]

    print("========== FINAL RESNET-18 ==========")
    print(
        "Selected epoch:",
        verified["selected_epoch"],
    )
    print(
        "Fit splits:",
        verified["fit_data"]["splits"],
    )
    print(
        "Fit records:",
        verified["fit_data"]["records"],
    )
    print(
        "Trainable parameters:",
        verified[
            "model_contract"
        ]["trainable_parameters"],
    )
    print(
        "Final train losses:",
        verified[
            "fit_result"
        ]["train_losses"],
    )
    print()
    print(
        "Test records:",
        verified["test_data"]["records"],
    )
    print(
        "Test loss:",
        f"{verified['test_data']['loss']:.4f}",
    )
    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}",
    )
    print(
        "Balanced accuracy:",
        f"{metrics['balanced_accuracy']:.4f}",
    )
    print(
        "Precision:",
        f"{metrics['precision']:.4f}",
    )
    print(
        "Recall:",
        f"{metrics['recall']:.4f}",
    )
    print(
        "F1:",
        f"{metrics['f1']:.4f}",
    )
    print(
        "ROC-AUC:",
        f"{metrics['roc_auc']:.4f}",
    )
    print(
        "Confusion matrix:",
        metrics["confusion_matrix"],
    )
    print()
    print(
        "Model:",
        model_path.relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Evaluation report:",
        report_path.relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Predictions:",
        predictions_path.relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print()
    print(
        "PASS: locked ResNet-18 was refitted "
        "on train plus validation."
    )
    print(
        "PASS: untouched test split was "
        "evaluated once."
    )
    print(
        "Use --verify-only for all later checks."
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
