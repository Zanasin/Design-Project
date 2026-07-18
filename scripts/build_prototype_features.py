from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data import (
    ImagePreprocessor,
    load_manifest,
    records_for_split,
)
from src.features import (
    HandcraftedFeatureConfig,
    build_split_feature_cache,
    sha256_file,
    verify_feature_cache_metadata,
    write_json_atomic,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_features.yaml"
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping in {path}"
        )

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic handcrafted feature "
            "caches for the Phase 2 prototype."
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
        help="Replace existing feature caches.",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing cache files without rebuilding.",
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(config_path)

    feature_config = (
        HandcraftedFeatureConfig.from_mapping(
            config["feature_contract"]
        )
    )

    manifest_path = (
        PROJECT_ROOT
        / config["source"]["manifest"]
    ).resolve()

    output_root = (
        PROJECT_ROOT
        / config["cache"]["output_root"]
    ).resolve()

    metadata_path = (
        PROJECT_ROOT
        / config["cache"]["metadata"]
    ).resolve()

    resolved_project_root = PROJECT_ROOT.resolve()

    if not output_root.is_relative_to(
        resolved_project_root
    ):
        raise ValueError(
            "Feature output root must remain inside "
            "the project directory"
        )

    manifest_sha256 = sha256_file(
        manifest_path
    )

    feature_config_sha256 = sha256_file(
        config_path
    )

    if args.verify_only:
        metadata = verify_feature_cache_metadata(
            metadata_path,
            resolved_project_root,
            expected_manifest_sha256=(
                manifest_sha256
            ),
            expected_config_sha256=(
                feature_config_sha256
            ),
        )

        print("Feature cache verification passed.")
        print(
            "Cached records:",
            sum(
                int(value["record_count"])
                for value in metadata[
                    "splits"
                ].values()
            ),
        )

        return 0

    records = load_manifest(
        manifest_path
    )

    input_config = config[
        "feature_contract"
    ]["input"]

    preprocessor = ImagePreprocessor(
        width=int(input_config["width"]),
        height=int(input_config["height"]),
        colour_mode=str(
            input_config["colour_mode"]
        ),
    )

    split_metadata: dict[str, Any] = {}

    for split in config["cache"]["splits"]:
        selected_records = records_for_split(
            records,
            split,
        )

        split_metadata[split] = (
            build_split_feature_cache(
                records=selected_records,
                project_root=(
                    resolved_project_root
                ),
                output_root=output_root,
                preprocessor=preprocessor,
                feature_config=feature_config,
                split=split,
                force=args.force,
                show_progress=True,
            )
        )

    total_bytes = sum(
        int(file_metadata["bytes"])
        for split in split_metadata.values()
        for file_metadata in split[
            "files"
        ].values()
    )

    metadata = {
        "cache_name": (
            "CIFAKE Phase 2 handcrafted features"
        ),
        "cache_version": 1,
        "deterministic": True,
        "manifest": config["source"]["manifest"],
        "manifest_sha256": manifest_sha256,
        "feature_config": config_path.relative_to(
            resolved_project_root
        ).as_posix(),
        "feature_config_sha256": (
            feature_config_sha256
        ),
        "output_root": config[
            "cache"
        ]["output_root"],
        "feature_dtype": config[
            "cache"
        ]["feature_dtype"],
        "label_dtype": config[
            "cache"
        ]["label_dtype"],
        "dimensions": (
            feature_config.dimensions
        ),
        "combined_order": list(
            feature_config.combined_order
        ),
        "total_records": len(records),
        "total_cache_bytes": total_bytes,
        "splits": split_metadata,
    }

    write_json_atomic(
        metadata_path,
        metadata,
    )

    verify_feature_cache_metadata(
        metadata_path,
        resolved_project_root,
        expected_manifest_sha256=(
            manifest_sha256
        ),
        expected_config_sha256=(
            feature_config_sha256
        ),
    )

    print()
    print("========== FEATURE CACHE ==========")

    for split, details in split_metadata.items():
        print(
            f"{split:10}",
            details["record_count"],
            details["class_counts"],
        )

    print()
    print("Dimensions:")

    for name, dimension in (
        feature_config.dimensions.items()
    ):
        print(f"  {name:8}: {dimension}")

    print(
        "Cache size:",
        f"{total_bytes / (1024 ** 2):.2f} MiB",
    )

    print(
        "Metadata:",
        metadata_path.relative_to(
            resolved_project_root
        ),
    )

    print()
    print(
        "Prototype feature cache completed "
        "and verified."
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
