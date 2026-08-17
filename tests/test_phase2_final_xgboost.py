from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.classical import (
    XGBoostSelectionConfig,
    evaluate_final_xgboost,
    fit_final_xgboost,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTION_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_xgboost.yaml"
)

SELECTION_REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "prototype"
    / "xgboost"
    / "validation_selection.json"
)

FINAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_xgboost_final.yaml"
)


def load_selection_config() -> (
    XGBoostSelectionConfig
):
    data = yaml.safe_load(
        SELECTION_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    return XGBoostSelectionConfig.from_mapping(
        data["xgboost_selection"]
    )


def locked_candidate():
    config = load_selection_config()

    return next(
        candidate
        for candidate in config.candidates
        if candidate.name == "slower_learning"
    )


def synthetic_splits():
    features, labels = make_classification(
        n_samples=240,
        n_features=30,
        n_informative=18,
        n_redundant=6,
        class_sep=1.3,
        random_state=42,
    )

    (
        development_features,
        test_features,
        development_labels,
        test_labels,
    ) = train_test_split(
        features.astype(np.float32),
        labels.astype(np.int64),
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
    data = yaml.safe_load(
        FINAL_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    final = data["final_xgboost"]

    assert final["development_splits"] == [
        "train",
        "validation",
    ]

    assert final["test_split"] == "test"

    assert "test" not in final[
        "development_splits"
    ]

    paths = list(
        final["artifacts"].values()
    )

    assert len(paths) == len(set(paths))


def test_locked_candidate_matches_report() -> None:
    report = json.loads(
        SELECTION_REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    candidate = locked_candidate()

    assert report[
        "selected_candidate"
    ] == candidate.name

    assert report["selected_parameters"] == {
        "learning_rate": (
            candidate.learning_rate
        ),
        "max_depth": candidate.max_depth,
        "n_estimators": (
            candidate.n_estimators
        ),
    }


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
    candidate = locked_candidate()

    first_fit = fit_final_xgboost(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        candidate=candidate,
        config=config,
    )

    second_fit = fit_final_xgboost(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        candidate=candidate,
        config=config,
    )

    assert first_fit.record_count == (
        train_features.shape[0]
        + validation_features.shape[0]
    )

    assert (
        first_fit.boosted_rounds
        == candidate.n_estimators
    )

    first_evaluation = (
        evaluate_final_xgboost(
            model=first_fit.model,
            test_features=test_features,
            test_labels=test_labels,
        )
    )

    second_evaluation = (
        evaluate_final_xgboost(
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
        first_evaluation.positive_probabilities,
        second_evaluation.positive_probabilities,
    )

    assert (
        first_evaluation.metrics
        == second_evaluation.metrics
    )


def test_test_dimension_mismatch_is_rejected() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    ) = synthetic_splits()

    fit = fit_final_xgboost(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        candidate=locked_candidate(),
        config=load_selection_config(),
    )

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        evaluate_final_xgboost(
            model=fit.model,
            test_features=test_features[:, :-1],
            test_labels=test_labels,
        )
