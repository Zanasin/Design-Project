from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.features import (
    sha256_file,
    verify_feature_cache_metadata,
)
from src.models.classical import (
    LinearSVMSelectionConfig,
    evaluate_final_linear_svm,
    fit_final_linear_svm,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_svm_final.yaml"
)

CLASS_NAMES = {
    0: "real",
    1: "ai_generated",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping in {path}"
        )

    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return data


def resolve_project_path(
    relative_path: str,
) -> Path:
    project_root = PROJECT_ROOT.resolve()

    path = (
        project_root / relative_path
    ).resolve()

    if not path.is_relative_to(project_root):
        raise ValueError(
            f"Path escapes project root: {relative_path}"
        )

    return path


def load_cached_array(
    *,
    metadata: dict[str, Any],
    cache_root: Path,
    split: str,
    name: str,
) -> np.ndarray:
    relative_path = Path(
        metadata[
            "splits"
        ][split]["files"][name]["path"]
    )

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError(
            f"Unsafe cache path for {split}/{name}"
        )

    path = (
        cache_root / relative_path
    ).resolve()

    if not path.is_relative_to(cache_root):
        raise ValueError(
            f"Cache path escapes root for {split}/{name}"
        )

    return np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )


def validate_locked_selection(
    *,
    selection_path: Path,
    selection: dict[str, Any],
    metadata_path: Path,
    metadata: dict[str, Any],
) -> tuple[
    LinearSVMSelectionConfig,
    float,
    Path,
]:
    if selection["model"] != "linear_svm":
        raise ValueError(
            "Locked selection is not a Linear SVM"
        )

    selection_config_path = resolve_project_path(
        selection["svm_config"]
    )

    if (
        sha256_file(selection_config_path)
        != selection["svm_config_sha256"]
    ):
        raise ValueError(
            "Locked SVM configuration hash changed"
        )

    if (
        sha256_file(metadata_path)
        != selection[
            "feature_cache_metadata_sha256"
        ]
    ):
        raise ValueError(
            "Feature-cache metadata hash changed "
            "after model selection"
        )

    if (
        selection["manifest_sha256"]
        != metadata["manifest_sha256"]
    ):
        raise ValueError(
            "Manifest hash no longer matches selection"
        )

    if (
        selection["feature_config_sha256"]
        != metadata["feature_config_sha256"]
    ):
        raise ValueError(
            "Feature configuration hash no longer "
            "matches selection"
        )

    selection_config_file = load_yaml(
        selection_config_path
    )

    selection_config = (
        LinearSVMSelectionConfig.from_mapping(
            selection_config_file[
                "linear_svm_selection"
            ]
        )
    )

    selected_c = float(
        selection["selected_c"]
    )

    selected_candidate = next(
        (
            candidate
            for candidate in selection[
                "candidates"
            ]
            if float(candidate["c"])
            == selected_c
        ),
        None,
    )

    if selected_candidate is None:
        raise ValueError(
            "Selected C is absent from candidate results"
        )

    if not selected_candidate["converged"]:
        raise ValueError(
            "Selected validation candidate did not converge"
        )

    if (
        selection["development_data"]["fit_split"]
        != "train"
        or selection["development_data"][
            "selection_split"
        ]
        != "validation"
    ):
        raise ValueError(
            "Locked selection did not use the required "
            "development splits"
        )

    return (
        selection_config,
        selected_c,
        selection_config_path,
    )


def temporary_path(path: Path) -> Path:
    return path.with_name(
        path.name + ".temporary"
    )


def write_predictions(
    *,
    path: Path,
    image_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
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
                "decision_score",
                "correct",
            ],
            lineterminator="\n",
        )

        writer.writeheader()

        for (
            image_id,
            true_label,
            predicted_label,
            decision_score,
        ) in zip(
            image_ids,
            labels,
            predictions,
            decision_scores,
            strict=True,
        ):
            true_value = int(true_label)
            predicted_value = int(
                predicted_label
            )

            writer.writerow(
                {
                    "image_id": str(image_id),
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
                    "decision_score": format(
                        float(decision_score),
                        ".17g",
                    ),
                    "correct": int(
                        true_value
                        == predicted_value
                    ),
                }
            )


def write_json(
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


def verify_final_artifacts(
    *,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    final_config = config[
        "final_linear_svm"
    ]

    selection_path = resolve_project_path(
        final_config["selection_report"]
    )

    metadata_path = resolve_project_path(
        final_config[
            "feature_cache_metadata"
        ]
    )

    model_path = resolve_project_path(
        final_config["artifacts"]["model"]
    )

    report_path = resolve_project_path(
        final_config[
            "artifacts"
        ]["test_evaluation"]
    )

    predictions_path = resolve_project_path(
        final_config[
            "artifacts"
        ]["test_predictions"]
    )

    for path in [
        selection_path,
        metadata_path,
        model_path,
        report_path,
        predictions_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact is missing: {path}"
            )

    report = load_json(report_path)

    if (
        report["final_config_sha256"]
        != sha256_file(config_path)
    ):
        raise ValueError(
            "Final configuration hash mismatch"
        )

    if (
        report["selection_report_sha256"]
        != sha256_file(selection_path)
    ):
        raise ValueError(
            "Selection report hash mismatch"
        )

    if (
        report[
            "feature_cache_metadata_sha256"
        ]
        != sha256_file(metadata_path)
    ):
        raise ValueError(
            "Feature-cache metadata hash mismatch"
        )

    if (
        report["artifacts"]["model"]["sha256"]
        != sha256_file(model_path)
    ):
        raise ValueError(
            "Saved model hash mismatch"
        )

    if (
        report[
            "artifacts"
        ]["test_predictions"]["sha256"]
        != sha256_file(predictions_path)
    ):
        raise ValueError(
            "Prediction file hash mismatch"
        )

    model = joblib.load(model_path)

    saved_c = float(
        model.named_steps["classifier"].C
    )

    if saved_c != float(report["selected_c"]):
        raise ValueError(
            "Saved model C does not match report"
        )

    with predictions_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        prediction_rows = list(
            csv.DictReader(file)
        )

    expected_records = int(
        report["test_data"]["records"]
    )

    if len(prediction_rows) != expected_records:
        raise ValueError(
            "Prediction row count does not match report"
        )

    confusion_total = sum(
        int(value)
        for row in report[
            "test_metrics"
        ]["confusion_matrix"]
        for value in row
    )

    if confusion_total != expected_records:
        raise ValueError(
            "Confusion matrix count does not match "
            "test records"
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the locked prototype Linear SVM on "
            "train plus validation and evaluate the "
            "untouched test split once."
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
            "Verify existing final artifacts without "
            "refitting or evaluating the test split."
        ),
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(config_path)

    final_config = config[
        "final_linear_svm"
    ]

    development_splits = list(
        final_config["development_splits"]
    )

    test_split = str(
        final_config["test_split"]
    )

    if development_splits != [
        "train",
        "validation",
    ]:
        raise ValueError(
            "Final development splits must be "
            "train and validation"
        )

    if test_split != "test":
        raise ValueError(
            "Final evaluation split must be test"
        )

    if test_split in development_splits:
        raise ValueError(
            "Test split cannot be part of model fitting"
        )

    if args.verify_only:
        report = verify_final_artifacts(
            config_path=config_path,
            config=config,
        )

        print(
            "Final Linear SVM artifact "
            "verification passed."
        )
        print(
            "Selected C:",
            report["selected_c"],
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
        final_config["selection_report"]
    )

    metadata_path = resolve_project_path(
        final_config[
            "feature_cache_metadata"
        ]
    )

    model_path = resolve_project_path(
        final_config["artifacts"]["model"]
    )

    report_path = resolve_project_path(
        final_config[
            "artifacts"
        ]["test_evaluation"]
    )

    predictions_path = resolve_project_path(
        final_config[
            "artifacts"
        ]["test_predictions"]
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
            "Final artifacts already exist. "
            "Use --verify-only instead: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    selection = load_json(
        selection_path
    )

    metadata = load_json(
        metadata_path
    )

    (
        selection_config,
        selected_c,
        selection_config_path,
    ) = validate_locked_selection(
        selection_path=selection_path,
        selection=selection,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    manifest_path = resolve_project_path(
        metadata["manifest"]
    )

    feature_config_path = resolve_project_path(
        metadata["feature_config"]
    )

    verify_feature_cache_metadata(
        metadata_path,
        PROJECT_ROOT.resolve(),
        expected_manifest_sha256=(
            sha256_file(manifest_path)
        ),
        expected_config_sha256=(
            sha256_file(feature_config_path)
        ),
    )

    cache_root = resolve_project_path(
        metadata["output_root"]
    )

    feature_name = str(
        final_config["feature_name"]
    )

    arrays: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for split in [
        *development_splits,
        test_split,
    ]:
        arrays[split] = {
            "features": load_cached_array(
                metadata=metadata,
                cache_root=cache_root,
                split=split,
                name=feature_name,
            ),
            "labels": load_cached_array(
                metadata=metadata,
                cache_root=cache_root,
                split=split,
                name="labels",
            ),
            "image_ids": load_cached_array(
                metadata=metadata,
                cache_root=cache_root,
                split=split,
                name="image_ids",
            ),
        }

    split_id_sets = {
        split: set(
            values["image_ids"].tolist()
        )
        for split, values in arrays.items()
    }

    overlaps = {
        "train_validation": len(
            split_id_sets["train"]
            & split_id_sets["validation"]
        ),
        "train_test": len(
            split_id_sets["train"]
            & split_id_sets["test"]
        ),
        "validation_test": len(
            split_id_sets["validation"]
            & split_id_sets["test"]
        ),
    }

    if any(overlaps.values()):
        raise ValueError(
            f"Split image-ID overlap found: {overlaps}"
        )

    fit_result = fit_final_linear_svm(
        train_features=arrays[
            "train"
        ]["features"],
        train_labels=arrays[
            "train"
        ]["labels"],
        validation_features=arrays[
            "validation"
        ]["features"],
        validation_labels=arrays[
            "validation"
        ]["labels"],
        selected_c=selected_c,
        config=selection_config,
    )

    if not fit_result.converged:
        raise RuntimeError(
            "Final Linear SVM did not converge"
        )

    evaluation = evaluate_final_linear_svm(
        model=fit_result.model,
        test_features=arrays[
            test_split
        ]["features"],
        test_labels=arrays[
            test_split
        ]["labels"],
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

        joblib.dump(
            fit_result.model,
            model_temporary,
            compress=int(
                final_config[
                    "joblib_compress"
                ]
            ),
        )

        write_predictions(
            path=predictions_temporary,
            image_ids=arrays[
                test_split
            ]["image_ids"],
            labels=arrays[
                test_split
            ]["labels"],
            predictions=(
                evaluation.predictions
            ),
            decision_scores=(
                evaluation.decision_scores
            ),
        )

        fit_labels = np.concatenate(
            [
                arrays["train"]["labels"],
                arrays["validation"][
                    "labels"
                ],
            ]
        )

        test_labels = arrays[
            test_split
        ]["labels"]

        report = {
            "schema_version": 1,
            "model": "linear_svm",
            "kernel_interpretation": "linear",
            "selected_c": selected_c,
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
            "feature_cache_metadata": (
                metadata_path.relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix()
            ),
            "feature_cache_metadata_sha256": (
                sha256_file(metadata_path)
            ),
            "manifest_sha256": (
                metadata["manifest_sha256"]
            ),
            "feature_config_sha256": (
                metadata[
                    "feature_config_sha256"
                ]
            ),
            "feature_name": feature_name,
            "feature_dimension": (
                fit_result.feature_dimension
            ),
            "fit_data": {
                "splits": development_splits,
                "records": (
                    fit_result.record_count
                ),
                "class_counts": {
                    str(label): count
                    for label, count in sorted(
                        Counter(
                            int(value)
                            for value in fit_labels
                        ).items()
                    )
                },
                "split_image_id_overlaps": (
                    overlaps
                ),
            },
            "fit_result": {
                "converged": (
                    fit_result.converged
                ),
                "n_iter": fit_result.n_iter,
            },
            "test_data": {
                "split": test_split,
                "records": int(
                    test_labels.shape[0]
                ),
                "class_counts": {
                    str(label): count
                    for label, count in sorted(
                        Counter(
                            int(value)
                            for value in test_labels
                        ).items()
                    )
                },
            },
            "test_metrics": (
                evaluation.metrics
            ),
            "software": {
                "python": (
                    platform.python_version()
                ),
                "numpy": np.__version__,
                "scikit_learn": (
                    sklearn.__version__
                ),
                "joblib": joblib.__version__,
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
                    "sha256": sha256_file(
                        model_temporary
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
                    "sha256": sha256_file(
                        predictions_temporary
                    ),
                },
            },
        }

        write_json(
            report_temporary,
            report,
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.parent.mkdir(
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

    verified_report = verify_final_artifacts(
        config_path=config_path,
        config=config,
    )

    metrics = verified_report[
        "test_metrics"
    ]

    print(
        "========== FINAL LINEAR SVM =========="
    )
    print(
        "Locked C:",
        verified_report["selected_c"],
    )
    print(
        "Fit splits:",
        verified_report[
            "fit_data"
        ]["splits"],
    )
    print(
        "Fit records:",
        verified_report[
            "fit_data"
        ]["records"],
    )
    print(
        "Feature dimension:",
        verified_report[
            "feature_dimension"
        ],
    )
    print(
        "Converged:",
        verified_report[
            "fit_result"
        ]["converged"],
    )
    print(
        "Iterations:",
        verified_report[
            "fit_result"
        ]["n_iter"],
    )
    print()
    print(
        "Test split:",
        verified_report[
            "test_data"
        ]["split"],
    )
    print(
        "Test records:",
        verified_report[
            "test_data"
        ]["records"],
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
        "PASS: locked Linear SVM was refitted "
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
