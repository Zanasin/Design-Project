from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SUPPORTED_SELECTION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
}


@dataclass(frozen=True, slots=True)
class LinearSVMSelectionConfig:
    random_state: int
    candidate_c_values: tuple[float, ...]
    selection_metric: str

    scaler_with_mean: bool
    scaler_with_std: bool

    classifier_dual: str | bool
    classifier_max_iter: int
    classifier_tolerance: float
    classifier_class_weight: str | None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> LinearSVMSelectionConfig:
        scaler = mapping["scaler"]
        classifier = mapping["classifier"]

        config = cls(
            random_state=int(mapping["random_state"]),
            candidate_c_values=tuple(
                float(value)
                for value in mapping[
                    "candidate_c_values"
                ]
            ),
            selection_metric=str(
                mapping["selection_metric"]
            ),
            scaler_with_mean=bool(
                scaler["with_mean"]
            ),
            scaler_with_std=bool(
                scaler["with_std"]
            ),
            classifier_dual=classifier["dual"],
            classifier_max_iter=int(
                classifier["max_iter"]
            ),
            classifier_tolerance=float(
                classifier["tolerance"]
            ),
            classifier_class_weight=(
                classifier["class_weight"]
            ),
        )

        config.validate()
        return config

    def validate(self) -> None:
        if not self.candidate_c_values:
            raise ValueError(
                "At least one C candidate is required"
            )

        if any(
            value <= 0.0
            for value in self.candidate_c_values
        ):
            raise ValueError(
                "All C candidates must be positive"
            )

        if len(
            set(self.candidate_c_values)
        ) != len(self.candidate_c_values):
            raise ValueError(
                "C candidates must be unique"
            )

        if tuple(
            sorted(self.candidate_c_values)
        ) != self.candidate_c_values:
            raise ValueError(
                "C candidates must be sorted"
            )

        if (
            self.selection_metric
            not in SUPPORTED_SELECTION_METRICS
        ):
            raise ValueError(
                "Unsupported SVM selection metric: "
                f"{self.selection_metric}"
            )

        if self.classifier_max_iter <= 0:
            raise ValueError(
                "max_iter must be positive"
            )

        if self.classifier_tolerance <= 0.0:
            raise ValueError(
                "Tolerance must be positive"
            )


def build_linear_svm_pipeline(
    *,
    c_value: float,
    config: LinearSVMSelectionConfig,
) -> Pipeline:
    if c_value <= 0.0:
        raise ValueError("C must be positive")

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(
                    with_mean=(
                        config.scaler_with_mean
                    ),
                    with_std=(
                        config.scaler_with_std
                    ),
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=float(c_value),
                    dual=config.classifier_dual,
                    class_weight=(
                        config.classifier_class_weight
                    ),
                    random_state=(
                        config.random_state
                    ),
                    max_iter=(
                        config.classifier_max_iter
                    ),
                    tol=(
                        config.classifier_tolerance
                    ),
                ),
            ),
        ]
    )


def validate_binary_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    split_name: str,
) -> None:
    if features.ndim != 2:
        raise ValueError(
            f"{split_name} features must be two-dimensional"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"{split_name} labels must be one-dimensional"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split_name} feature and label counts differ"
        )

    if features.shape[0] == 0:
        raise ValueError(
            f"{split_name} split is empty"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{split_name} features contain non-finite values"
        )

    unique_labels = set(
        int(value)
        for value in np.unique(labels)
    )

    if unique_labels != {0, 1}:
        raise ValueError(
            f"{split_name} must contain labels 0 and 1"
        )


def calculate_binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
) -> dict[str, Any]:
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                decision_scores,
            )
        ),
        "confusion_matrix": [
            [
                int(value)
                for value in row
            ]
            for row in matrix.tolist()
        ],
    }


def choose_best_candidate(
    candidates: Sequence[dict[str, Any]],
    *,
    selection_metric: str,
) -> dict[str, Any]:
    converged_candidates = [
        candidate
        for candidate in candidates
        if candidate["converged"]
    ]

    if not converged_candidates:
        raise RuntimeError(
            "No Linear SVM candidate converged"
        )

    return max(
        converged_candidates,
        key=lambda candidate: (
            candidate["metrics"][
                selection_metric
            ],
            candidate["metrics"]["roc_auc"],
            -float(candidate["c"]),
        ),
    )


def select_linear_svm(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: LinearSVMSelectionConfig,
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

    candidates: list[dict[str, Any]] = []

    for c_value in config.candidate_c_values:
        pipeline = build_linear_svm_pipeline(
            c_value=c_value,
            config=config,
        )

        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            pipeline.fit(
                train_features,
                train_labels,
            )

        convergence_warnings = [
            warning
            for warning in caught
            if issubclass(
                warning.category,
                ConvergenceWarning,
            )
        ]

        predictions = pipeline.predict(
            validation_features
        )

        decision_scores = (
            pipeline.decision_function(
                validation_features
            )
        )

        classifier = pipeline.named_steps[
            "classifier"
        ]

        iteration_count = int(
            np.max(
                np.atleast_1d(
                    classifier.n_iter_
                )
            )
        )

        candidates.append(
            {
                "c": float(c_value),
                "converged": (
                    len(convergence_warnings) == 0
                ),
                "n_iter": iteration_count,
                "metrics": (
                    calculate_binary_metrics(
                        validation_labels,
                        predictions,
                        decision_scores,
                    )
                ),
            }
        )

    selected = choose_best_candidate(
        candidates,
        selection_metric=(
            config.selection_metric
        ),
    )

    return {
        "selection_metric": (
            config.selection_metric
        ),
        "selected_c": float(selected["c"]),
        "selected_validation_metrics": (
            selected["metrics"]
        ),
        "candidates": candidates,
    }
