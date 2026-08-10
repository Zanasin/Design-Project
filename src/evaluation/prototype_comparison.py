from __future__ import annotations

import csv
import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.features import sha256_file
from src.models.classical.linear_svm import (
    calculate_binary_metrics,
)


METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)

REQUIRED_PREDICTION_COLUMNS = {
    "image_id",
    "true_label",
    "predicted_label",
    "correct",
}


def resolve_project_path(
    project_root: Path,
    relative_path: str,
) -> Path:
    root = project_root.resolve()
    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
        raise ValueError(
            f"Path escapes project root: {relative_path}"
        )

    return path


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {path}"
        )

    return data


def load_prediction_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    missing = (
        REQUIRED_PREDICTION_COLUMNS
        - columns
    )

    if missing:
        raise ValueError(
            f"Missing prediction columns in {path}: "
            + ", ".join(sorted(missing))
        )

    return rows


def validate_and_summarize_model(
    *,
    model_config: dict[str, Any],
    project_root: Path,
    expected_records: int,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    model_key = str(model_config["key"])
    display_name = str(
        model_config["display_name"]
    )

    report_path = resolve_project_path(
        project_root,
        str(model_config["report"]),
    )

    predictions_path = resolve_project_path(
        project_root,
        str(model_config["predictions"]),
    )

    artifact_path = resolve_project_path(
        project_root,
        str(model_config["artifact"]),
    )

    for path in [
        report_path,
        predictions_path,
        artifact_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required source artifact is missing: {path}"
            )

    report = load_json(report_path)
    rows = load_prediction_rows(
        predictions_path
    )

    score_column = str(
        model_config["score_column"]
    )

    score_kind = str(
        model_config["score_kind"]
    )

    if not rows:
        raise ValueError(
            f"{display_name} prediction file is empty"
        )

    if score_column not in rows[0]:
        raise ValueError(
            f"Missing score column for "
            f"{display_name}: {score_column}"
        )

    if score_kind not in {
        "probability",
        "decision_function",
    }:
        raise ValueError(
            f"Unsupported score kind for "
            f"{display_name}: {score_kind}"
        )

    if (
        int(report["test_data"]["records"])
        != expected_records
    ):
        raise ValueError(
            f"{display_name} report has an "
            "unexpected test-record count"
        )

    if len(rows) != expected_records:
        raise ValueError(
            f"{display_name} prediction-row count "
            "does not match the comparison contract"
        )

    rows_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []

    for row in rows:
        image_id = str(row["image_id"])
        true_label = int(row["true_label"])
        predicted_label = int(
            row["predicted_label"]
        )
        score = float(
            row[score_column]
        )
        correct = int(row["correct"])

        if image_id in rows_by_id:
            raise ValueError(
                f"Duplicate image ID for "
                f"{display_name}: {image_id}"
            )

        if true_label not in {0, 1}:
            raise ValueError(
                f"Unexpected true label for {display_name}"
            )

        if predicted_label not in {0, 1}:
            raise ValueError(
                f"Unexpected predicted label for "
                f"{display_name}"
            )

        if not np.isfinite(score):
            raise ValueError(
                f"Non-finite score for "
                f"{display_name}"
            )

        if (
            score_kind == "probability"
            and not 0.0 <= score <= 1.0
        ):
            raise ValueError(
                f"Probability outside [0, 1] for "
                f"{display_name}"
            )

        expected_correct = int(
            true_label == predicted_label
        )

        if correct != expected_correct:
            raise ValueError(
                f"Incorrect correctness flag for "
                f"{display_name}/{image_id}"
            )

        rows_by_id[image_id] = {
            "true_label": true_label,
            "predicted_label": (
                predicted_label
            ),
            "score": score,
            "correct": bool(correct),
        }

        labels.append(true_label)
        predictions.append(predicted_label)
        scores.append(score)

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    predictions_array = np.asarray(
        predictions,
        dtype=np.int64,
    )

    scores_array = np.asarray(
        scores,
        dtype=np.float64,
    )

    recalculated = calculate_binary_metrics(
        labels_array,
        predictions_array,
        scores_array,
    )

    recorded = report["test_metrics"]

    for metric in METRIC_KEYS:
        if not np.isclose(
            float(recalculated[metric]),
            float(recorded[metric]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{display_name} {metric} differs "
                "between its report and predictions"
            )

    if (
        recalculated["confusion_matrix"]
        != recorded["confusion_matrix"]
    ):
        raise ValueError(
            f"{display_name} confusion matrix differs "
            "between its report and predictions"
        )

    confusion = recorded[
        "confusion_matrix"
    ]

    true_negative = int(confusion[0][0])
    false_positive = int(confusion[0][1])
    false_negative = int(confusion[1][0])
    true_positive = int(confusion[1][1])

    specificity_denominator = (
        true_negative + false_positive
    )

    specificity = (
        true_negative
        / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    artifact_bytes = (
        artifact_path.stat().st_size
    )

    selected_validation = report.get(
        "selected_validation_metrics",
        {},
    )

    summary = {
        "key": model_key,
        "display_name": display_name,
        "score_column": score_column,
        "score_kind": score_kind,
        "test_records": expected_records,
        "correct_predictions": int(
            predictions_array
            .__eq__(labels_array)
            .sum()
        ),
        "metrics": {
            metric: float(recorded[metric])
            for metric in METRIC_KEYS
        },
        "specificity": float(specificity),
        "confusion_matrix": confusion,
        "confusion_counts": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "selected_validation_metrics": {
            metric: float(
                selected_validation[metric]
            )
            for metric in METRIC_KEYS
            if metric in selected_validation
        },
        "sources": {
            "report": {
                "path": (
                    report_path.relative_to(
                        project_root.resolve()
                    ).as_posix()
                ),
                "sha256": sha256_file(
                    report_path
                ),
            },
            "predictions": {
                "path": (
                    predictions_path.relative_to(
                        project_root.resolve()
                    ).as_posix()
                ),
                "sha256": sha256_file(
                    predictions_path
                ),
            },
            "artifact": {
                "path": (
                    artifact_path.relative_to(
                        project_root.resolve()
                    ).as_posix()
                ),
                "sha256": sha256_file(
                    artifact_path
                ),
                "bytes": artifact_bytes,
                "mib": (
                    artifact_bytes
                    / 1024**2
                ),
            },
        },
    }

    return summary, rows_by_id


def build_prototype_comparison(
    *,
    config_file: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    config = config_file[
        "prototype_comparison"
    ]

    expected_records = int(
        config["expected_test_records"]
    )

    model_configs = list(
        config["models"]
    )

    if len(model_configs) != 3:
        raise ValueError(
            "Prototype comparison requires "
            "exactly three models"
        )

    model_keys = [
        str(item["key"])
        for item in model_configs
    ]

    display_names = [
        str(item["display_name"])
        for item in model_configs
    ]

    if len(set(model_keys)) != len(
        model_keys
    ):
        raise ValueError(
            "Model keys must be unique"
        )

    if len(set(display_names)) != len(
        display_names
    ):
        raise ValueError(
            "Model display names must be unique"
        )

    model_summaries: list[
        dict[str, Any]
    ] = []

    rows_by_model: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for model_config in model_configs:
        summary, rows_by_id = (
            validate_and_summarize_model(
                model_config=model_config,
                project_root=project_root,
                expected_records=(
                    expected_records
                ),
            )
        )

        model_summaries.append(summary)

        rows_by_model[
            summary["key"]
        ] = rows_by_id

    reference_key = model_keys[0]
    reference_rows = rows_by_model[
        reference_key
    ]

    reference_ids = set(
        reference_rows
    )

    reference_labels = {
        image_id: row["true_label"]
        for image_id, row
        in reference_rows.items()
    }

    for model_key in model_keys:
        rows = rows_by_model[model_key]

        if set(rows) != reference_ids:
            raise ValueError(
                f"{model_key} does not use the "
                "same test image IDs"
            )

        labels = {
            image_id: row["true_label"]
            for image_id, row
            in rows.items()
        }

        if labels != reference_labels:
            raise ValueError(
                f"{model_key} does not use the "
                "same true labels"
            )

    metric_keys = [
        str(metric)
        for metric in config["metrics"]
    ]

    if tuple(metric_keys) != METRIC_KEYS:
        raise ValueError(
            "Unexpected comparison metric contract"
        )

    metric_winners: dict[
        str,
        dict[str, Any],
    ] = {}

    for metric in metric_keys:
        best_value = max(
            model["metrics"][metric]
            for model in model_summaries
        )

        winners = [
            model["display_name"]
            for model in model_summaries
            if np.isclose(
                model["metrics"][metric],
                best_value,
                rtol=0.0,
                atol=1e-15,
            )
        ]

        metric_winners[metric] = {
            "models": winners,
            "value": float(best_value),
        }

    all_agree = 0
    disagreements = 0
    all_correct = 0
    all_wrong = 0

    only_model_correct = {
        model_key: 0
        for model_key in model_keys
    }

    pairwise_agreement = {
        f"{left}__{right}": 0
        for left, right
        in combinations(model_keys, 2)
    }

    for image_id in sorted(
        reference_ids
    ):
        predicted = {
            model_key: rows_by_model[
                model_key
            ][image_id]["predicted_label"]
            for model_key in model_keys
        }

        true_label = reference_labels[
            image_id
        ]

        correctness = {
            model_key: (
                predicted[model_key]
                == true_label
            )
            for model_key in model_keys
        }

        if len(set(predicted.values())) == 1:
            all_agree += 1
        else:
            disagreements += 1

        if all(correctness.values()):
            all_correct += 1

        if not any(correctness.values()):
            all_wrong += 1

        correct_models = [
            model_key
            for model_key, is_correct
            in correctness.items()
            if is_correct
        ]

        if len(correct_models) == 1:
            only_model_correct[
                correct_models[0]
            ] += 1

        for left, right in combinations(
            model_keys,
            2,
        ):
            if (
                predicted[left]
                == predicted[right]
            ):
                pairwise_agreement[
                    f"{left}__{right}"
                ] += 1

    smallest_model = min(
        model_summaries,
        key=lambda model: model[
            "sources"
        ]["artifact"]["bytes"],
    )

    f1_winner = metric_winners["f1"][
        "models"
    ][0]

    recall_winner = metric_winners[
        "recall"
    ]["models"][0]

    return {
        "schema_version": int(
            config["schema_version"]
        ),
        "scope": {
            "stage": "phase_2_prototype",
            "test_records": expected_records,
            "class_mapping": {
                str(key): value
                for key, value
                in config[
                    "class_mapping"
                ].items()
            },
            "identical_test_ids": True,
            "identical_true_labels": True,
        },
        "models": model_summaries,
        "metric_winners": metric_winners,
        "agreement": {
            "all_three_agree": all_agree,
            "at_least_one_disagreement": (
                disagreements
            ),
            "all_three_correct": all_correct,
            "all_three_wrong": all_wrong,
            "only_model_correct": (
                only_model_correct
            ),
            "pairwise_prediction_agreement": (
                pairwise_agreement
            ),
        },
        "summary": {
            "best_overall_by_f1": f1_winner,
            "best_recall": recall_winner,
            "smallest_artifact": (
                smallest_model[
                    "display_name"
                ]
            ),
        },
        "limitations": list(
            config["limitations"]
        ),
    }


def render_comparison_json(
    comparison: dict[str, Any],
) -> str:
    return (
        json.dumps(
            comparison,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_metrics_csv(
    comparison: dict[str, Any],
) -> str:
    output = io.StringIO(
        newline=""
    )

    fieldnames = [
        "model_key",
        "display_name",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "correct_predictions",
        "test_records",
        "artifact_bytes",
        "artifact_mib",
        "validation_f1",
        "validation_roc_auc",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )

    writer.writeheader()

    for model in comparison["models"]:
        metrics = model["metrics"]
        confusion = model[
            "confusion_counts"
        ]

        validation = model[
            "selected_validation_metrics"
        ]

        writer.writerow(
            {
                "model_key": model["key"],
                "display_name": (
                    model["display_name"]
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
                "specificity": format(
                    model["specificity"],
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
                "true_negative": confusion[
                    "true_negative"
                ],
                "false_positive": confusion[
                    "false_positive"
                ],
                "false_negative": confusion[
                    "false_negative"
                ],
                "true_positive": confusion[
                    "true_positive"
                ],
                "correct_predictions": model[
                    "correct_predictions"
                ],
                "test_records": model[
                    "test_records"
                ],
                "artifact_bytes": model[
                    "sources"
                ]["artifact"]["bytes"],
                "artifact_mib": format(
                    model["sources"][
                        "artifact"
                    ]["mib"],
                    ".17g",
                ),
                "validation_f1": format(
                    validation.get(
                        "f1",
                        float("nan"),
                    ),
                    ".17g",
                ),
                "validation_roc_auc": format(
                    validation.get(
                        "roc_auc",
                        float("nan"),
                    ),
                    ".17g",
                ),
            }
        )

    return output.getvalue()


def render_markdown_summary(
    comparison: dict[str, Any],
) -> str:
    models = comparison["models"]
    winners = comparison[
        "metric_winners"
    ]
    agreement = comparison["agreement"]
    summary = comparison["summary"]

    lines = [
        "# Prototype Model Comparison",
        "",
        "## Scope",
        "",
        (
            "All three finalized models were "
            "evaluated on the same 150 test images "
            "with identical labels: 75 real and "
            "75 AI-generated."
        ),
        "",
        "## Test metrics",
        "",
        (
            "| Model | Accuracy | Balanced accuracy "
            "| Precision | Recall | Specificity | "
            "F1 | ROC-AUC |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for model in models:
        metrics = model["metrics"]

        lines.append(
            "| "
            + model["display_name"]
            + " | "
            + f"{metrics['accuracy']:.4f}"
            + " | "
            + f"{metrics['balanced_accuracy']:.4f}"
            + " | "
            + f"{metrics['precision']:.4f}"
            + " | "
            + f"{metrics['recall']:.4f}"
            + " | "
            + f"{model['specificity']:.4f}"
            + " | "
            + f"{metrics['f1']:.4f}"
            + " | "
            + f"{metrics['roc_auc']:.4f}"
            + " |"
        )

    lines.extend(
        [
            "",
            "## Confusion matrices",
            "",
            (
                "| Model | TN | FP | FN | TP | "
                "Correct |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for model in models:
        counts = model[
            "confusion_counts"
        ]

        lines.append(
            "| "
            + model["display_name"]
            + " | "
            + str(counts["true_negative"])
            + " | "
            + str(counts["false_positive"])
            + " | "
            + str(counts["false_negative"])
            + " | "
            + str(counts["true_positive"])
            + " | "
            + str(model["correct_predictions"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## Metric leaders",
            "",
        ]
    )

    for metric in METRIC_KEYS:
        winner = winners[metric]

        lines.append(
            "- "
            + metric.replace("_", " ").title()
            + ": "
            + ", ".join(winner["models"])
            + f" ({winner['value']:.4f})"
        )

    lines.extend(
        [
            "",
            "## Model agreement",
            "",
            (
                f"- All three agree: "
                f"{agreement['all_three_agree']} / 150"
            ),
            (
                f"- At least one model disagrees: "
                f"{agreement['at_least_one_disagreement']} "
                "/ 150"
            ),
            (
                f"- All three correct: "
                f"{agreement['all_three_correct']} / 150"
            ),
            (
                f"- All three wrong: "
                f"{agreement['all_three_wrong']} / 150"
            ),
        ]
    )

    key_to_name = {
        model["key"]: model["display_name"]
        for model in models
    }

    for key, count in agreement[
        "only_model_correct"
    ].items():
        lines.append(
            f"- Only {key_to_name[key]} correct: "
            f"{count}"
        )

    lines.extend(
        [
            "",
            "## Artifact sizes",
            "",
            "| Model | Bytes | MiB |",
            "|---|---:|---:|",
        ]
    )

    for model in models:
        artifact = model[
            "sources"
        ]["artifact"]

        lines.append(
            "| "
            + model["display_name"]
            + " | "
            + f"{artifact['bytes']:,}"
            + " | "
            + f"{artifact['mib']:.2f}"
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- {summary['best_overall_by_f1']} "
                "is the strongest overall prototype "
                "according to F1."
            ),
            (
                f"- {summary['best_recall']} has the "
                "highest AI-image recall."
            ),
            (
                f"- {summary['smallest_artifact']} "
                "has the smallest saved model artifact."
            ),
            (
                "- The disagreement results show that "
                "the models learn partially different "
                "decision patterns."
            ),
            (
                "- Images misclassified by all three "
                "models should be prioritized during "
                "later failure analysis."
            ),
            "",
            "## Limitations",
            "",
        ]
    )

    for limitation in comparison[
        "limitations"
    ]:
        lines.append(
            "- " + str(limitation)
        )

    return "\n".join(lines) + "\n"
