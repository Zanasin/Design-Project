from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from xgboost import XGBClassifier

from src.models.classical.linear_svm import (
    SUPPORTED_SELECTION_METRICS,
    calculate_binary_metrics,
    validate_binary_arrays,
)


@dataclass(frozen=True, slots=True)
class XGBoostCandidateConfig:
    name: str
    n_estimators: int
    max_depth: int
    learning_rate: float

    @property
    def complexity(self) -> int:
        return self.n_estimators * self.max_depth

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> XGBoostCandidateConfig:
        candidate = cls(
            name=str(mapping["name"]),
            n_estimators=int(
                mapping["n_estimators"]
            ),
            max_depth=int(
                mapping["max_depth"]
            ),
            learning_rate=float(
                mapping["learning_rate"]
            ),
        )

        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.name:
            raise ValueError(
                "XGBoost candidate name cannot be empty"
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be positive"
            )

        if self.max_depth <= 0:
            raise ValueError(
                "max_depth must be positive"
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive"
            )


@dataclass(frozen=True, slots=True)
class XGBoostSelectionConfig:
    random_state: int
    selection_metric: str

    objective: str
    eval_metric: str
    tree_method: str
    device: str
    n_jobs: int

    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    min_child_weight: float
    gamma: float
    max_bin: int
    verbosity: int

    candidates: tuple[
        XGBoostCandidateConfig,
        ...,
    ]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> XGBoostSelectionConfig:
        fixed = mapping["fixed_parameters"]

        config = cls(
            random_state=int(
                mapping["random_state"]
            ),
            selection_metric=str(
                mapping["selection_metric"]
            ),
            objective=str(fixed["objective"]),
            eval_metric=str(
                fixed["eval_metric"]
            ),
            tree_method=str(
                fixed["tree_method"]
            ),
            device=str(fixed["device"]),
            n_jobs=int(fixed["n_jobs"]),
            subsample=float(
                fixed["subsample"]
            ),
            colsample_bytree=float(
                fixed["colsample_bytree"]
            ),
            reg_alpha=float(
                fixed["reg_alpha"]
            ),
            reg_lambda=float(
                fixed["reg_lambda"]
            ),
            min_child_weight=float(
                fixed["min_child_weight"]
            ),
            gamma=float(fixed["gamma"]),
            max_bin=int(fixed["max_bin"]),
            verbosity=int(
                fixed["verbosity"]
            ),
            candidates=tuple(
                XGBoostCandidateConfig.from_mapping(
                    candidate
                )
                for candidate in mapping[
                    "candidates"
                ]
            ),
        )

        config.validate()
        return config

    def validate(self) -> None:
        if (
            self.selection_metric
            not in SUPPORTED_SELECTION_METRICS
        ):
            raise ValueError(
                "Unsupported selection metric: "
                f"{self.selection_metric}"
            )

        if self.objective != "binary:logistic":
            raise ValueError(
                "Prototype XGBoost objective must be "
                "binary:logistic"
            )

        if self.tree_method != "hist":
            raise ValueError(
                "Prototype XGBoost tree method must be hist"
            )

        if self.device != "cpu":
            raise ValueError(
                "Prototype XGBoost selection must use CPU"
            )

        if self.n_jobs != 1:
            raise ValueError(
                "Deterministic selection requires n_jobs=1"
            )

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(
                "subsample must be in (0, 1]"
            )

        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError(
                "colsample_bytree must be in (0, 1]"
            )

        if self.reg_alpha < 0.0:
            raise ValueError(
                "reg_alpha cannot be negative"
            )

        if self.reg_lambda < 0.0:
            raise ValueError(
                "reg_lambda cannot be negative"
            )

        if self.min_child_weight < 0.0:
            raise ValueError(
                "min_child_weight cannot be negative"
            )

        if self.gamma < 0.0:
            raise ValueError(
                "gamma cannot be negative"
            )

        if self.max_bin < 2:
            raise ValueError(
                "max_bin must be at least 2"
            )

        if not self.candidates:
            raise ValueError(
                "At least one XGBoost candidate is required"
            )

        names = [
            candidate.name
            for candidate in self.candidates
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "XGBoost candidate names must be unique"
            )

        parameter_sets = [
            (
                candidate.n_estimators,
                candidate.max_depth,
                candidate.learning_rate,
            )
            for candidate in self.candidates
        ]

        if len(parameter_sets) != len(
            set(parameter_sets)
        ):
            raise ValueError(
                "XGBoost candidate parameter sets "
                "must be unique"
            )


def build_xgboost_classifier(
    *,
    candidate: XGBoostCandidateConfig,
    config: XGBoostSelectionConfig,
) -> XGBClassifier:
    return XGBClassifier(
        objective=config.objective,
        eval_metric=config.eval_metric,
        n_estimators=candidate.n_estimators,
        max_depth=candidate.max_depth,
        learning_rate=candidate.learning_rate,
        tree_method=config.tree_method,
        device=config.device,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        subsample=config.subsample,
        colsample_bytree=(
            config.colsample_bytree
        ),
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        min_child_weight=(
            config.min_child_weight
        ),
        gamma=config.gamma,
        max_bin=config.max_bin,
        verbosity=config.verbosity,
    )


def choose_best_xgboost_candidate(
    candidates: Sequence[dict[str, Any]],
    *,
    selection_metric: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError(
            "No XGBoost candidates were supplied"
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"][
                selection_metric
            ],
            candidate["metrics"]["roc_auc"],
            -int(candidate["complexity"]),
            -int(
                candidate["candidate_index"]
            ),
        ),
    )


def select_xgboost(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: XGBoostSelectionConfig,
) -> dict[str, Any]:
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

    results: list[dict[str, Any]] = []

    for index, candidate in enumerate(
        config.candidates
    ):
        model = build_xgboost_classifier(
            candidate=candidate,
            config=config,
        )

        model.fit(
            train_features,
            train_labels,
        )

        predictions = model.predict(
            validation_features
        ).astype(
            np.int64,
            copy=False,
        )

        probabilities = model.predict_proba(
            validation_features
        )

        if probabilities.shape != (
            validation_features.shape[0],
            2,
        ):
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
            validation_labels,
            predictions,
            positive_probabilities,
        )

        results.append(
            {
                "candidate_index": index,
                "name": candidate.name,
                "parameters": {
                    "n_estimators": (
                        candidate.n_estimators
                    ),
                    "max_depth": (
                        candidate.max_depth
                    ),
                    "learning_rate": (
                        candidate.learning_rate
                    ),
                },
                "complexity": (
                    candidate.complexity
                ),
                "boosted_rounds": int(
                    model.get_booster(
                    ).num_boosted_rounds()
                ),
                "nonzero_feature_importances": int(
                    np.count_nonzero(
                        model.feature_importances_
                    )
                ),
                "metrics": metrics,
            }
        )

    selected = (
        choose_best_xgboost_candidate(
            results,
            selection_metric=(
                config.selection_metric
            ),
        )
    )

    return {
        "selection_metric": (
            config.selection_metric
        ),
        "selected_candidate": (
            selected["name"]
        ),
        "selected_parameters": (
            selected["parameters"]
        ),
        "selected_validation_metrics": (
            selected["metrics"]
        ),
        "candidates": results,
    }
