from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.classical import (
    LinearSVMSelectionConfig,
    build_linear_svm_pipeline,
    calculate_binary_metrics,
    choose_best_candidate,
    select_linear_svm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_svm.yaml"
)


def load_config() -> LinearSVMSelectionConfig:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return LinearSVMSelectionConfig.from_mapping(
        config["linear_svm_selection"]
    )


def test_svm_config_and_pipeline_contract() -> None:
    config = load_config()

    assert config.candidate_c_values == (
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
    )

    assert config.selection_metric == "f1"

    pipeline = build_linear_svm_pipeline(
        c_value=0.1,
        config=config,
    )

    assert list(
        pipeline.named_steps
    ) == [
        "scaler",
        "classifier",
    ]

    assert pipeline["classifier"].C == 0.1
    assert (
        type(pipeline["classifier"]).__name__
        == "LinearSVC"
    )


def test_binary_metric_contract() -> None:
    labels = np.asarray(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    predictions = labels.copy()

    decision_scores = np.asarray(
        [-2.0, -1.0, 1.0, 2.0],
        dtype=np.float64,
    )

    metrics = calculate_binary_metrics(
        labels,
        predictions,
        decision_scores,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [
        [2, 0],
        [0, 2],
    ]


def test_lower_c_is_final_tie_breaker() -> None:
    candidates = [
        {
            "c": 1.0,
            "converged": True,
            "metrics": {
                "f1": 0.8,
                "roc_auc": 0.9,
            },
        },
        {
            "c": 0.1,
            "converged": True,
            "metrics": {
                "f1": 0.8,
                "roc_auc": 0.9,
            },
        },
    ]

    selected = choose_best_candidate(
        candidates,
        selection_metric="f1",
    )

    assert selected["c"] == 0.1


def test_svm_selection_is_deterministic() -> None:
    features, labels = make_classification(
        n_samples=180,
        n_features=24,
        n_informative=14,
        n_redundant=4,
        n_classes=2,
        class_sep=1.2,
        random_state=42,
    )

    (
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    ) = train_test_split(
        features.astype(np.float32),
        labels.astype(np.int64),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    config = load_config()

    first = select_linear_svm(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=(
            validation_features
        ),
        validation_labels=(
            validation_labels
        ),
        config=config,
    )

    second = select_linear_svm(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=(
            validation_features
        ),
        validation_labels=(
            validation_labels
        ),
        config=config,
    )

    assert first == second
    assert first["selected_c"] in (
        config.candidate_c_values
    )


def test_selection_config_excludes_test_split() -> None:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    selection = config[
        "linear_svm_selection"
    ]

    assert selection["fit_split"] == "train"
    assert (
        selection["selection_split"]
        == "validation"
    )

    assert "test" not in {
        selection["fit_split"],
        selection["selection_split"],
    }
