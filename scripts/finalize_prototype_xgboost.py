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
import sklearn
import xgboost
import yaml
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.features import (
    sha256_file,
    verify_feature_cache_metadata,
)
from src.models.classical import (
    XGBoostCandidateConfig,
    XGBoostSelectionConfig,
    evaluate_final_xgboost,
    fit_final_xgboost,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_xgboost_final.yaml"
)

CLASS_NAMES = {
    0: "real",
    1: "ai_generated",
}


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
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
    root = PROJECT_ROOT.resolve()
    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
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
        metadata["splits"][split][
            "files"
        ][name]["path"]
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


def temporary_path(path: Path) -> Path:
    return path.with_name(
        path.stem
        + ".temporary"
        + path.suffix
    )


def write_predictions(
    *,
    path: Path,
    image_ids: np.ndarray,
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


def validate_locked_selection(
    *,
    selection: dict[str, Any],
    metadata_path: Path,
    metadata: dict[str, Any],
) -> tuple[
    XGBoostSelectionConfig,
    XGBoostCandidateConfig,
    Path,
]:
    if selection["model"] != "xgboost":
        raise ValueError(
            "Locked selection is not XGBoost"
        )

    config_path = resolve_project_path(
        selection["xgboost_config"]
    )

    if (
        sha256_file(config_path)
        != selection["xgboost_config_sha256"]
    ):
        raise ValueError(
            "Locked XGBoost configuration changed"
        )

    if (
        sha256_file(metadata_path)
        != selection[
            "feature_cache_metadata_sha256"
        ]
    ):
        raise ValueError(
            "Feature-cache metadata changed "
            "after XGBoost selection"
        )

    if (
        selection["manifest_sha256"]
        != metadata["manifest_sha256"]
    ):
        raise ValueError(
            "Manifest hash mismatch"
        )

    if (
        selection["feature_config_sha256"]
        != metadata["feature_config_sha256"]
    ):
        raise ValueError(
            "Feature configuration hash mismatch"
        )

    development = selection[
        "development_data"
    ]

    if (
        development["fit_split"] != "train"
        or development["selection_split"]
        != "validation"
    ):
        raise ValueError(
            "Locked XGBoost selection used "
            "unexpected development splits"
        )

    config_file = load_yaml(config_path)

    model_config = (
        XGBoostSelectionConfig.from_mapping(
            config_file[
                "xgboost_selection"
            ]
        )
    )

    selected_name = str(
        selection["selected_candidate"]
    )

    candidate = next(
        (
            item
            for item in model_config.candidates
            if item.name == selected_name
        ),
        None,
    )

    if candidate is None:
        raise ValueError(
            "Selected candidate is absent "
            "from the locked configuration"
        )

    expected_parameters = {
        "learning_rate": (
            candidate.learning_rate
        ),
        "max_depth": candidate.max_depth,
        "n_estimators": (
            candidate.n_estimators
        ),
    }

    if (
        selection["selected_parameters"]
        != expected_parameters
    ):
        raise ValueError(
            "Selected parameters do not match "
            "the locked candidate"
        )

    return (
        model_config,
        candidate,
        config_path,
    )


def verify_final_artifacts(
    *,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    final = config["final_xgboost"]

    selection_path = resolve_project_path(
        final["selection_report"]
    )

    metadata_path = resolve_project_path(
        final["feature_cache_metadata"]
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
            report[
                "feature_cache_metadata_sha256"
            ],
            sha256_file(metadata_path),
            "feature-cache metadata",
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

    model = XGBClassifier()
    model.load_model(model_path)

    rounds = int(
        model.get_booster().num_boosted_rounds()
    )

    expected_rounds = int(
        report["selected_parameters"][
            "n_estimators"
        ]
    )

    if rounds != expected_rounds:
        raise ValueError(
            "Saved XGBoost round count mismatch"
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

    confusion_total = sum(
        int(value)
        for row in report[
            "test_metrics"
        ]["confusion_matrix"]
        for value in row
    )

    if confusion_total != expected_records:
        raise ValueError(
            "Confusion matrix total mismatch"
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the locked prototype XGBoost "
            "model on train plus validation and "
            "evaluate the untouched test split once."
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
            "refitting or evaluating the test split."
        ),
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    final = config["final_xgboost"]

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
            "Test cannot be used for fitting"
        )

    if args.verify_only:
        report = verify_final_artifacts(
            config_path=config_path,
            config=config,
        )

        print(
            "Final XGBoost artifact "
            "verification passed."
        )
        print(
            "Candidate:",
            report["selected_candidate"],
        )
        print(
            "Parameters:",
            report["selected_parameters"],
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

    metadata_path = resolve_project_path(
        final["feature_cache_metadata"]
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
            "Final XGBoost artifacts already "
            "exist. Use --verify-only instead: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    selection = load_json(selection_path)
    metadata = load_json(metadata_path)

    (
        model_config,
        candidate,
        selection_config_path,
    ) = validate_locked_selection(
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
        final["feature_name"]
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

    id_sets = {
        split: set(
            values["image_ids"].tolist()
        )
        for split, values in arrays.items()
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
            f"Split image-ID overlap found: {overlaps}"
        )

    fit = fit_final_xgboost(
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
        candidate=candidate,
        config=model_config,
    )

    evaluation = evaluate_final_xgboost(
        model=fit.model,
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

        fit.model.save_model(
            model_temporary
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
            probabilities=(
                evaluation.positive_probabilities
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
            "model": "xgboost",
            "selected_candidate": (
                candidate.name
            ),
            "selected_parameters": {
                "learning_rate": (
                    candidate.learning_rate
                ),
                "max_depth": (
                    candidate.max_depth
                ),
                "n_estimators": (
                    candidate.n_estimators
                ),
            },
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
                fit.feature_dimension
            ),
            "fit_data": {
                "splits": development_splits,
                "records": fit.record_count,
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
                "boosted_rounds": (
                    fit.boosted_rounds
                ),
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
                "xgboost": (
                    xgboost.__version__
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

    verified = verify_final_artifacts(
        config_path=config_path,
        config=config,
    )

    metrics = verified["test_metrics"]

    print("========== FINAL XGBOOST ==========")
    print(
        "Candidate:",
        verified["selected_candidate"],
    )
    print(
        "Parameters:",
        verified["selected_parameters"],
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
        "Feature dimension:",
        verified["feature_dimension"],
    )
    print(
        "Boosted rounds:",
        verified[
            "fit_result"
        ]["boosted_rounds"],
    )
    print()
    print(
        "Test records:",
        verified["test_data"]["records"],
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
        "PASS: locked XGBoost was refitted "
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
