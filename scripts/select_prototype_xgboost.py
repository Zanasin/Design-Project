from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.features import (
    sha256_file,
    verify_feature_cache_metadata,
    write_json_atomic,
)
from src.models.classical import (
    XGBoostSelectionConfig,
    select_xgboost,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_xgboost.yaml"
)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping in {path}"
        )

    return data


def resolve_project_path(
    relative_path: str,
) -> Path:
    root = PROJECT_ROOT.resolve()
    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
        raise ValueError(
            f"Path escapes project root: {relative_path}"
        )

    return path


def load_cached_array(
    *,
    metadata: dict[str, Any],
    cache_root: Path,
    split: str,
    name: str,
) -> np.ndarray:
    relative_path = Path(
        metadata["splits"][split][
            "files"
        ][name]["path"]
    )

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError(
            f"Unsafe cache path for {split}/{name}"
        )

    path = (
        cache_root / relative_path
    ).resolve()

    if not path.is_relative_to(cache_root):
        raise ValueError(
            f"Cache path escapes root for {split}/{name}"
        )

    return np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select prototype XGBoost parameters "
            "using train and validation only."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing selection report.",
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config_file = load_yaml(config_path)

    selection_mapping = config_file[
        "xgboost_selection"
    ]

    fit_split = str(
        selection_mapping["fit_split"]
    )

    selection_split = str(
        selection_mapping["selection_split"]
    )

    if (
        fit_split != "train"
        or selection_split != "validation"
    ):
        raise ValueError(
            "Phase 2F-2 requires train for fitting "
            "and validation for selection"
        )

    if "test" in {
        fit_split,
        selection_split,
    }:
        raise ValueError(
            "Test split cannot be used for selection"
        )

    model_config = (
        XGBoostSelectionConfig.from_mapping(
            selection_mapping
        )
    )

    metadata_path = resolve_project_path(
        config_file["source"][
            "feature_cache_metadata"
        ]
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    manifest_path = resolve_project_path(
        metadata["manifest"]
    )

    feature_config_path = resolve_project_path(
        metadata["feature_config"]
    )

    verify_feature_cache_metadata(
        metadata_path,
        PROJECT_ROOT.resolve(),
        expected_manifest_sha256=(
            sha256_file(manifest_path)
        ),
        expected_config_sha256=(
            sha256_file(feature_config_path)
        ),
    )

    cache_root = resolve_project_path(
        metadata["output_root"]
    )

    feature_name = str(
        selection_mapping["feature_name"]
    )

    train_features = load_cached_array(
        metadata=metadata,
        cache_root=cache_root,
        split=fit_split,
        name=feature_name,
    )

    train_labels = load_cached_array(
        metadata=metadata,
        cache_root=cache_root,
        split=fit_split,
        name="labels",
    )

    train_ids = load_cached_array(
        metadata=metadata,
        cache_root=cache_root,
        split=fit_split,
        name="image_ids",
    )

    validation_features = load_cached_array(
        metadata=metadata,
        cache_root=cache_root,
        split=selection_split,
        name=feature_name,
    )

    validation_labels = load_cached_array(
        metadata=metadata,
        cache_root=cache_root,
        split=selection_split,
        name="labels",
    )

    validation_ids = load_cached_array(
        metadata=metadata,
        cache_root=cache_root,
        split=selection_split,
        name="image_ids",
    )

    overlap = set(
        train_ids.tolist()
    ) & set(
        validation_ids.tolist()
    )

    if overlap:
        raise ValueError(
            "Train and validation image IDs overlap"
        )

    result = select_xgboost(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=(
            validation_features
        ),
        validation_labels=(
            validation_labels
        ),
        config=model_config,
    )

    report_path = resolve_project_path(
        config_file["output"][
            "selection_report"
        ]
    )

    if report_path.exists() and not args.force:
        raise FileExistsError(
            f"Selection report already exists: "
            f"{report_path}. Use --force to replace it."
        )

    report = {
        "schema_version": 1,
        "model": "xgboost",
        "random_state": (
            model_config.random_state
        ),
        "xgboost_version_contract": {
            "objective": (
                model_config.objective
            ),
            "eval_metric": (
                model_config.eval_metric
            ),
            "tree_method": (
                model_config.tree_method
            ),
            "device": model_config.device,
            "n_jobs": model_config.n_jobs,
        },
        "xgboost_config": (
            config_path.relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        ),
        "xgboost_config_sha256": (
            sha256_file(config_path)
        ),
        "feature_cache_metadata": (
            metadata_path.relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        ),
        "feature_cache_metadata_sha256": (
            sha256_file(metadata_path)
        ),
        "manifest_sha256": (
            metadata["manifest_sha256"]
        ),
        "feature_config_sha256": (
            metadata["feature_config_sha256"]
        ),
        "development_data": {
            "feature_name": feature_name,
            "feature_dimension": int(
                train_features.shape[1]
            ),
            "fit_split": fit_split,
            "fit_records": int(
                train_features.shape[0]
            ),
            "fit_class_counts": {
                str(label): count
                for label, count in sorted(
                    Counter(
                        int(value)
                        for value in train_labels
                    ).items()
                )
            },
            "selection_split": (
                selection_split
            ),
            "selection_records": int(
                validation_features.shape[0]
            ),
            "selection_class_counts": {
                str(label): count
                for label, count in sorted(
                    Counter(
                        int(value)
                        for value
                        in validation_labels
                    ).items()
                )
            },
            "image_id_overlap": len(overlap),
        },
        "selection_rule": {
            "primary_metric": (
                model_config.selection_metric
            ),
            "tie_breakers": list(
                selection_mapping[
                    "tie_breakers"
                ]
            ),
        },
        **result,
    }

    write_json_atomic(
        report_path,
        report,
    )

    print(
        "========== XGBOOST VALIDATION =========="
    )
    print(
        "Fit split:",
        fit_split,
        train_features.shape,
    )
    print(
        "Selection split:",
        selection_split,
        validation_features.shape,
    )
    print("Test split loaded: NO")
    print()

    print(
        f"{'Candidate':<28}"
        f"{'Trees':>7}"
        f"{'Depth':>7}"
        f"{'Rate':>9}"
        f"{'F1':>9}"
        f"{'ROC-AUC':>10}"
        f"{'Accuracy':>10}"
    )

    for candidate in result["candidates"]:
        parameters = candidate[
            "parameters"
        ]

        metrics = candidate["metrics"]

        print(
            f"{candidate['name']:<28}"
            f"{parameters['n_estimators']:>7}"
            f"{parameters['max_depth']:>7}"
            f"{parameters['learning_rate']:>9.3f}"
            f"{metrics['f1']:>9.4f}"
            f"{metrics['roc_auc']:>10.4f}"
            f"{metrics['accuracy']:>10.4f}"
        )

    print()
    print(
        "Selected candidate:",
        result["selected_candidate"],
    )
    print(
        "Selected parameters:",
        result["selected_parameters"],
    )
    print(
        "Selected validation F1:",
        f"{result['selected_validation_metrics']['f1']:.4f}",
    )
    print(
        "Selected validation ROC-AUC:",
        f"{result['selected_validation_metrics']['roc_auc']:.4f}",
    )
    print(
        "Report:",
        report_path.relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Report SHA-256:",
        sha256_file(report_path),
    )
    print()
    print(
        "PASS: XGBoost selected using "
        "train and validation only."
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
