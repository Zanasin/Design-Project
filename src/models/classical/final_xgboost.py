from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.models.classical.final_linear_svm import (
    combine_development_splits,
)
from src.models.classical.linear_svm import (
    calculate_binary_metrics,
    validate_binary_arrays,
)
from src.models.classical.xgboost_model import (
    XGBoostCandidateConfig,
    XGBoostSelectionConfig,
    build_xgboost_classifier,
)


@dataclass(slots=True)
class FinalXGBoostFit:
    model: XGBClassifier
    record_count: int
    feature_dimension: int
    boosted_rounds: int


@dataclass(frozen=True, slots=True)
class FinalXGBoostEvaluation:
    predictions: np.ndarray
    positive_probabilities: np.ndarray
    metrics: dict[str, Any]


def fit_final_xgboost(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    candidate: XGBoostCandidateConfig,
    config: XGBoostSelectionConfig,
) -> FinalXGBoostFit:
    development_features, development_labels = (
        combine_development_splits(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )
    )

    model = build_xgboost_classifier(
        candidate=candidate,
        config=config,
    )

    model.fit(
        development_features,
        development_labels,
    )

    boosted_rounds = int(
        model.get_booster().num_boosted_rounds()
    )

    if boosted_rounds != candidate.n_estimators:
        raise RuntimeError(
            "Final XGBoost boosted-round count "
            "does not match the locked configuration"
        )

    return FinalXGBoostFit(
        model=model,
        record_count=int(
            development_features.shape[0]
        ),
        feature_dimension=int(
            development_features.shape[1]
        ),
        boosted_rounds=boosted_rounds,
    )


def evaluate_final_xgboost(
    *,
    model: XGBClassifier,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> FinalXGBoostEvaluation:
    validate_binary_arrays(
        test_features,
        test_labels,
        split_name="test",
    )

    expected_dimension = int(
        model.n_features_in_
    )

    if test_features.shape[1] != expected_dimension:
        raise ValueError(
            "Test feature dimension does not match "
            "the fitted XGBoost model"
        )

    predictions = model.predict(
        test_features
    ).astype(
        np.int64,
        copy=False,
    )

    probabilities = model.predict_proba(
        test_features
    )

    expected_shape = (
        test_features.shape[0],
        2,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "Unexpected XGBoost probability shape"
        )

    positive_probabilities = (
        probabilities[:, 1].astype(
            np.float64,
            copy=False,
        )
    )

    metrics = calculate_binary_metrics(
        test_labels,
        predictions,
        positive_probabilities,
    )

    return FinalXGBoostEvaluation(
        predictions=predictions,
        positive_probabilities=(
            positive_probabilities
        ),
        metrics=metrics,
    )
