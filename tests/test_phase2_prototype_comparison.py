from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.evaluation import (
    build_prototype_comparison,
    render_comparison_json,
    render_markdown_summary,
    render_metrics_csv,
)
from src.models.classical.linear_svm import (
    calculate_binary_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_comparison.yaml"
)


def write_synthetic_source(
    *,
    root: Path,
    key: str,
    predictions: list[int],
    probabilities: list[float],
) -> dict[str, str]:
    directory = root / key
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [0, 0, 1, 1]
    image_ids = [
        "image-0",
        "image-1",
        "image-2",
        "image-3",
    ]

    metrics = calculate_binary_metrics(
        labels,
        predictions,
        probabilities,
    )

    report_path = (
        directory / "report.json"
    )

    predictions_path = (
        directory / "predictions.csv"
    )

    artifact_path = (
        directory / "model.bin"
    )

    report = {
        "test_data": {
            "records": 4,
        },
        "test_metrics": metrics,
        "selected_validation_metrics": {
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        },
    }

    report_path.write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "true_label",
                "predicted_label",
                "positive_probability",
                "correct",
            ],
            lineterminator="\n",
        )

        writer.writeheader()

        for (
            image_id,
            label,
            prediction,
            probability,
        ) in zip(
            image_ids,
            labels,
            predictions,
            probabilities,
            strict=True,
        ):
            writer.writerow(
                {
                    "image_id": image_id,
                    "true_label": label,
                    "predicted_label": (
                        prediction
                    ),
                    "positive_probability": (
                        probability
                    ),
                    "correct": int(
                        label == prediction
                    ),
                }
            )

    artifact_path.write_bytes(
        key.encode("utf-8")
    )

    return {
        "report": (
            report_path.relative_to(root)
            .as_posix()
        ),
        "predictions": (
            predictions_path.relative_to(
                root
            ).as_posix()
        ),
        "artifact": (
            artifact_path.relative_to(root)
            .as_posix()
        ),
    }


def synthetic_config(
    tmp_path: Path,
):
    sources = {
        "model_a": write_synthetic_source(
            root=tmp_path,
            key="model_a",
            predictions=[0, 1, 1, 0],
            probabilities=[
                -2.0,
                1.0,
                2.0,
                -0.5,
            ],
        ),
        "model_b": write_synthetic_source(
            root=tmp_path,
            key="model_b",
            predictions=[0, 0, 1, 1],
            probabilities=[
                0.1,
                0.2,
                0.8,
                0.9,
            ],
        ),
        "model_c": write_synthetic_source(
            root=tmp_path,
            key="model_c",
            predictions=[1, 0, 1, 0],
            probabilities=[
                0.7,
                0.2,
                0.8,
                0.3,
            ],
        ),
    }

    model_a_predictions = (
        tmp_path
        / sources["model_a"]["predictions"]
    )

    model_a_content = (
        model_a_predictions.read_text(
            encoding="utf-8"
        )
    )

    model_a_predictions.write_text(
        model_a_content.replace(
            "positive_probability",
            "decision_score",
            1,
        ),
        encoding="utf-8",
    )

    models = []

    for key in [
        "model_a",
        "model_b",
        "model_c",
    ]:
        models.append(
            {
                "key": key,
                "display_name": key.upper(),
                "score_column": (
                    "decision_score"
                    if key == "model_a"
                    else "positive_probability"
                ),
                "score_kind": (
                    "decision_function"
                    if key == "model_a"
                    else "probability"
                ),
                **sources[key],
            }
        )

    return {
        "prototype_comparison": {
            "schema_version": 1,
            "expected_test_records": 4,
            "class_mapping": {
                0: "real",
                1: "ai_generated",
            },
            "models": models,
            "metrics": [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ],
            "limitations": [
                "Synthetic test limitation."
            ],
        }
    }


def test_real_configuration_contract() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )["prototype_comparison"]

    assert config[
        "expected_test_records"
    ] == 150

    assert [
        model["key"]
        for model in config["models"]
    ] == [
        "linear_svm",
        "xgboost",
        "resnet18",
    ]

    outputs = list(
        config["outputs"].values()
    )

    assert len(outputs) == len(
        set(outputs)
    )


def test_synthetic_sources_are_compared(
    tmp_path: Path,
) -> None:
    comparison = (
        build_prototype_comparison(
            config_file=synthetic_config(
                tmp_path
            ),
            project_root=tmp_path,
        )
    )

    assert comparison["scope"][
        "test_records"
    ] == 4

    assert comparison["scope"][
        "identical_test_ids"
    ] is True

    assert comparison["summary"][
        "best_overall_by_f1"
    ] == "MODEL_B"

    assert comparison["agreement"][
        "all_three_wrong"
    ] == 0


def test_synthetic_agreement_counts(
    tmp_path: Path,
) -> None:
    comparison = (
        build_prototype_comparison(
            config_file=synthetic_config(
                tmp_path
            ),
            project_root=tmp_path,
        )
    )

    agreement = comparison[
        "agreement"
    ]

    assert (
        agreement["all_three_agree"]
        + agreement[
            "at_least_one_disagreement"
        ]
        == 4
    )

    assert (
        agreement["all_three_correct"]
        <= 4
    )


def test_rendering_is_deterministic(
    tmp_path: Path,
) -> None:
    config = synthetic_config(
        tmp_path
    )

    first = build_prototype_comparison(
        config_file=config,
        project_root=tmp_path,
    )

    second = build_prototype_comparison(
        config_file=config,
        project_root=tmp_path,
    )

    assert render_comparison_json(
        first
    ) == render_comparison_json(second)

    assert render_metrics_csv(
        first
    ) == render_metrics_csv(second)

    assert render_markdown_summary(
        first
    ) == render_markdown_summary(second)
