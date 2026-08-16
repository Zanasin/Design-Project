from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.classical import (
    LinearSVMSelectionConfig,
    combine_development_splits,
    evaluate_final_linear_svm,
    fit_final_linear_svm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTION_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_svm.yaml"
)

FINAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_svm_final.yaml"
)


def load_selection_config() -> (
    LinearSVMSelectionConfig
):
    config = yaml.safe_load(
        SELECTION_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    return LinearSVMSelectionConfig.from_mapping(
        config["linear_svm_selection"]
    )


def synthetic_splits() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    features, labels = make_classification(
        n_samples=240,
        n_features=30,
        n_informative=18,
        n_redundant=6,
        class_sep=1.3,
        random_state=42,
    )

    features = features.astype(
        np.float32
    )

    labels = labels.astype(
        np.int64
    )

    (
        development_features,
        test_features,
        development_labels,
        test_labels,
    ) = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    (
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    ) = train_test_split(
        development_features,
        development_labels,
        test_size=0.25,
        random_state=42,
        stratify=development_labels,
    )

    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    )


def test_final_configuration_contract() -> None:
    config = yaml.safe_load(
        FINAL_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    final = config["final_linear_svm"]

    assert final["development_splits"] == [
        "train",
        "validation",
    ]

    assert final["test_split"] == "test"

    assert "test" not in final[
        "development_splits"
    ]

    artifact_paths = list(
        final["artifacts"].values()
    )

    assert len(artifact_paths) == len(
        set(artifact_paths)
    )


def test_development_split_combination() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        _,
        _,
    ) = synthetic_splits()

    features, labels = (
        combine_development_splits(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )
    )

    assert features.shape[0] == (
        train_features.shape[0]
        + validation_features.shape[0]
    )

    assert labels.shape[0] == features.shape[0]
    assert features.dtype == np.float32
    assert labels.dtype == np.int64


def test_final_fit_and_evaluation_are_deterministic() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    ) = synthetic_splits()

    config = load_selection_config()

    first_fit = fit_final_linear_svm(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        selected_c=0.0001,
        config=config,
    )

    second_fit = fit_final_linear_svm(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        selected_c=0.0001,
        config=config,
    )

    assert first_fit.converged
    assert second_fit.converged

    assert first_fit.record_count == (
        train_features.shape[0]
        + validation_features.shape[0]
    )

    assert (
        first_fit.model[
            "classifier"
        ].C
        == 0.0001
    )

    first_evaluation = (
        evaluate_final_linear_svm(
            model=first_fit.model,
            test_features=test_features,
            test_labels=test_labels,
        )
    )

    second_evaluation = (
        evaluate_final_linear_svm(
            model=second_fit.model,
            test_features=test_features,
            test_labels=test_labels,
        )
    )

    assert np.array_equal(
        first_evaluation.predictions,
        second_evaluation.predictions,
    )

    assert np.array_equal(
        first_evaluation.decision_scores,
        second_evaluation.decision_scores,
    )

    assert (
        first_evaluation.metrics
        == second_evaluation.metrics
    )

    confusion_total = sum(
        value
        for row in first_evaluation.metrics[
            "confusion_matrix"
        ]
        for value in row
    )

    assert confusion_total == test_labels.shape[0]


def test_test_dimension_mismatch_is_rejected() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    ) = synthetic_splits()

    fit = fit_final_linear_svm(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        selected_c=0.0001,
        config=load_selection_config(),
    )

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        evaluate_final_linear_svm(
            model=fit.model,
            test_features=test_features[:, :-1],
            test_labels=test_labels,
        )
