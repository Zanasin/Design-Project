from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.classical import (
    XGBoostSelectionConfig,
    build_xgboost_classifier,
    choose_best_xgboost_candidate,
    select_xgboost,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_xgboost.yaml"
)


def load_config() -> XGBoostSelectionConfig:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    return XGBoostSelectionConfig.from_mapping(
        config["xgboost_selection"]
    )


def test_xgboost_configuration_contract() -> None:
    config = load_config()

    assert config.random_state == 42
    assert config.selection_metric == "f1"
    assert config.tree_method == "hist"
    assert config.device == "cpu"
    assert config.n_jobs == 1

    assert [
        candidate.name
        for candidate in config.candidates
    ] == [
        "baseline",
        "shallow",
        "slower_learning",
        "shallow_slower_learning",
    ]

    baseline = config.candidates[0]

    assert baseline.n_estimators == 50
    assert baseline.max_depth == 3
    assert baseline.learning_rate == 0.1

    model = build_xgboost_classifier(
        candidate=baseline,
        config=config,
    )

    parameters = model.get_params()

    assert parameters["tree_method"] == "hist"
    assert parameters["device"] == "cpu"
    assert parameters["n_jobs"] == 1


def test_lower_complexity_is_tie_breaker() -> None:
    candidates = [
        {
            "candidate_index": 0,
            "complexity": 300,
            "metrics": {
                "f1": 0.8,
                "roc_auc": 0.9,
            },
        },
        {
            "candidate_index": 1,
            "complexity": 100,
            "metrics": {
                "f1": 0.8,
                "roc_auc": 0.9,
            },
        },
    ]

    selected = (
        choose_best_xgboost_candidate(
            candidates,
            selection_metric="f1",
        )
    )

    assert selected["complexity"] == 100


def test_xgboost_selection_is_deterministic() -> None:
    features, labels = make_classification(
        n_samples=180,
        n_features=24,
        n_informative=14,
        n_redundant=4,
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

    first = select_xgboost(
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

    second = select_xgboost(
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

    assert first["selected_candidate"] in {
        candidate.name
        for candidate in config.candidates
    }


def test_selection_configuration_excludes_test() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    selection = config[
        "xgboost_selection"
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

def test_feature_dimension_mismatch_is_rejected() -> None:
    train_features = np.zeros(
        (6, 4),
        dtype=np.float32,
    )

    validation_features = np.zeros(
        (6, 3),
        dtype=np.float32,
    )

    train_labels = np.asarray(
        [0, 1, 0, 1, 0, 1],
        dtype=np.int64,
    )

    validation_labels = train_labels.copy()

    with pytest.raises(
        ValueError,
        match=(
            "Train and validation feature "
            "dimensions differ"
        ),
    ):
        select_xgboost(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=(
                validation_features
            ),
            validation_labels=(
                validation_labels
            ),
            config=load_config(),
        )
