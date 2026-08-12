from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.models.classical.linear_svm import (
    LinearSVMSelectionConfig,
    build_linear_svm_pipeline,
    calculate_binary_metrics,
    validate_binary_arrays,
)


@dataclass(slots=True)
class FinalLinearSVMFit:
    model: Pipeline
    converged: bool
    n_iter: int
    record_count: int
    feature_dimension: int


@dataclass(frozen=True, slots=True)
class FinalLinearSVMEvaluation:
    predictions: np.ndarray
    decision_scores: np.ndarray
    metrics: dict[str, Any]


def combine_development_splits(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    validate_binary_arrays(
        train_features,
        train_labels,
        split_name="train",
    )

    validate_binary_arrays(
        validation_features,
        validation_labels,
        split_name="validation",
    )

    if (
        train_features.shape[1]
        != validation_features.shape[1]
    ):
        raise ValueError(
            "Train and validation feature dimensions differ"
        )

    features = np.concatenate(
        [
            train_features,
            validation_features,
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    labels = np.concatenate(
        [
            train_labels,
            validation_labels,
        ],
        axis=0,
    ).astype(
        np.int64,
        copy=False,
    )

    return features, labels


def fit_final_linear_svm(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    selected_c: float,
    config: LinearSVMSelectionConfig,
) -> FinalLinearSVMFit:
    development_features, development_labels = (
        combine_development_splits(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )
    )

    model = build_linear_svm_pipeline(
        c_value=selected_c,
        config=config,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            development_features,
            development_labels,
        )

    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    iteration_count = int(
        np.max(
            np.atleast_1d(
                classifier.n_iter_
            )
        )
    )

    return FinalLinearSVMFit(
        model=model,
        converged=(
            len(convergence_warnings) == 0
        ),
        n_iter=iteration_count,
        record_count=int(
            development_features.shape[0]
        ),
        feature_dimension=int(
            development_features.shape[1]
        ),
    )


def evaluate_final_linear_svm(
    *,
    model: Pipeline,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> FinalLinearSVMEvaluation:
    validate_binary_arrays(
        test_features,
        test_labels,
        split_name="test",
    )

    expected_features = int(
        model.named_steps[
            "classifier"
        ].n_features_in_
    )

    if test_features.shape[1] != expected_features:
        raise ValueError(
            "Test feature dimension does not match "
            "the fitted model"
        )

    predictions = model.predict(
        test_features
    ).astype(
        np.int64,
        copy=False,
    )

    decision_scores = model.decision_function(
        test_features
    ).astype(
        np.float64,
        copy=False,
    )

    metrics = calculate_binary_metrics(
        test_labels,
        predictions,
        decision_scores,
    )

    return FinalLinearSVMEvaluation(
        predictions=predictions,
        decision_scores=decision_scores,
        metrics=metrics,
    )
