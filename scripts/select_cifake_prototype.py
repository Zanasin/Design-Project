from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "prototype_selection.yaml"
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")

    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")

    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def path_selection_rank(
    relative_path: str,
    seed: int,
) -> str:
    value = f"{seed}:{relative_path}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def classify_source_path(
    path: Path,
    source_root: Path,
) -> tuple[str | None, str | None]:
    relative_parts = {
        part.lower()
        for part in path.relative_to(source_root).parts[:-1]
    }

    source_split: str | None = None
    source_label: str | None = None

    if "train" in relative_parts:
        source_split = "train"
    elif "test" in relative_parts:
        source_split = "test"

    if "real" in relative_parts:
        source_label = "real"
    elif "fake" in relative_parts:
        source_label = "fake"

    return source_split, source_label


def scan_source_images(
    source_root: Path,
) -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = {
        ("train", "real"): [],
        ("train", "fake"): [],
        ("test", "real"): [],
        ("test", "fake"): [],
    }

    unclassified: list[Path] = []

    for path in source_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        source_split, source_label = classify_source_path(
            path,
            source_root,
        )

        key = (source_split, source_label)

        if key not in groups:
            unclassified.append(path)
            continue

        groups[key].append(path)

    if unclassified:
        examples = "\n".join(
            str(path.relative_to(PROJECT_ROOT))
            for path in unclassified[:10]
        )

        raise RuntimeError(
            f"Found {len(unclassified)} unclassified images.\n"
            f"Examples:\n{examples}"
        )

    return groups


def choose_unique_images(
    candidates: list[Path],
    required_count: int,
    source_root: Path,
    seed: int,
    used_content_hashes: set[str],
) -> tuple[list[dict[str, Any]], int]:
    ranked_candidates = sorted(
        candidates,
        key=lambda path: path_selection_rank(
            path.relative_to(source_root).as_posix(),
            seed,
        ),
    )

    selected: list[dict[str, Any]] = []
    duplicate_candidates_skipped = 0

    for source_path in ranked_candidates:
        dataset_relative_path = (
            source_path.relative_to(source_root).as_posix()
        )

        content_hash = sha256_file(source_path)

        if content_hash in used_content_hashes:
            duplicate_candidates_skipped += 1
            continue

        used_content_hashes.add(content_hash)

        selected.append(
            {
                "source_path": source_path,
                "source_dataset_relative_path": (
                    dataset_relative_path
                ),
                "source_sha256": content_hash,
                "selection_rank": path_selection_rank(
                    dataset_relative_path,
                    seed,
                ),
            }
        )

        if len(selected) == required_count:
            break

    if len(selected) != required_count:
        raise RuntimeError(
            "Could not select enough unique images: "
            f"required {required_count}, found {len(selected)}"
        )

    return selected, duplicate_candidates_skipped


def prepare_output_directories(
    directories: list[Path],
    force: bool,
) -> None:
    existing_images: list[Path] = []

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

        existing_images.extend(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )

    if existing_images and not force:
        raise RuntimeError(
            "Prototype output directories already contain "
            f"{len(existing_images)} images. "
            "Use --force only when intentionally rebuilding "
            "the deterministic selection."
        )

    if force:
        for path in existing_images:
            path.unlink()


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "image_id",
        "split",
        "label",
        "label_id",
        "source_dataset",
        "source_version",
        "source_original_split",
        "source_original_label",
        "source_project_relative_path",
        "source_dataset_relative_path",
        "selected_relative_path",
        "source_sha256",
        "copied_sha256",
        "selection_rank",
        "file_size_bytes",
        "width",
        "height",
        "mode",
        "format",
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the deterministic 1,000-image "
            "CIFAKE Phase 2 prototype."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate existing selected images.",
    )

    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)

    selection_config = config["selection"]
    source_config = config["source"]
    output_config = config["outputs"]
    label_config = config["labels"]
    count_config = config["counts"]
    rule_config = config["rules"]

    seed = int(selection_config["seed"])

    source_metadata_path = (
        PROJECT_ROOT / source_config["metadata"]
    )

    source_metadata = load_json(source_metadata_path)

    if source_metadata["audit_passed"] is not True:
        raise RuntimeError(
            "CIFAKE source audit did not pass."
        )

    if (
        source_metadata["dataset_version"]
        != source_config["expected_version"]
    ):
        raise RuntimeError(
            "Unexpected CIFAKE dataset version."
        )

    source_root = (
        PROJECT_ROOT / source_metadata["download_root"]
    ).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_root}"
        )

    output_directories = {
        "real": (
            PROJECT_ROOT
            / output_config["real_directory"]
        ),
        "ai_generated": (
            PROJECT_ROOT
            / output_config["ai_generated_directory"]
        ),
    }

    prepare_output_directories(
        list(output_directories.values()),
        force=args.force,
    )

    print("========== SOURCE SCAN ==========")
    print("Source root:", source_root.relative_to(PROJECT_ROOT))

    groups = scan_source_images(source_root)

    for key in [
        ("train", "real"),
        ("train", "fake"),
        ("test", "real"),
        ("test", "fake"),
    ]:
        print(
            f"{key[0]}_{key[1]:5}",
            f"{len(groups[key]):>8,}",
        )

    expected_source_counts = {
        ("train", "real"): 50_000,
        ("train", "fake"): 50_000,
        ("test", "real"): 10_000,
        ("test", "fake"): 10_000,
    }

    for key, expected_count in expected_source_counts.items():
        if len(groups[key]) != expected_count:
            raise RuntimeError(
                f"Unexpected source count for {key}: "
                f"expected {expected_count}, "
                f"found {len(groups[key])}"
            )

    used_content_hashes: set[str] = set()
    duplicate_candidates_skipped = 0

    selected_groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for output_label in ["real", "ai_generated"]:
        source_label = label_config[
            output_label
        ]["source_label"]

        train_and_validation_count = (
            int(count_config["train"][output_label])
            + int(count_config["validation"][output_label])
        )

        train_pool, skipped = choose_unique_images(
            candidates=groups[("train", source_label)],
            required_count=train_and_validation_count,
            source_root=source_root,
            seed=seed,
            used_content_hashes=used_content_hashes,
        )

        duplicate_candidates_skipped += skipped

        train_count = int(
            count_config["train"][output_label]
        )

        selected_groups[("train", output_label)] = (
            train_pool[:train_count]
        )

        selected_groups[
            ("validation", output_label)
        ] = train_pool[train_count:]

        test_selection, skipped = choose_unique_images(
            candidates=groups[("test", source_label)],
            required_count=int(
                count_config["test"][output_label]
            ),
            source_root=source_root,
            seed=seed,
            used_content_hashes=used_content_hashes,
        )

        duplicate_candidates_skipped += skipped

        selected_groups[("test", output_label)] = (
            test_selection
        )

    rows: list[dict[str, Any]] = []
    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    for split in ["train", "validation", "test"]:
        expected_source_split = rule_config[
            f"{split}_source_split"
        ]

        for output_label in ["real", "ai_generated"]:
            label_id = int(
                label_config[output_label]["label_id"]
            )

            source_label = label_config[
                output_label
            ]["source_label"]

            selections = selected_groups[
                (split, output_label)
            ]

            for index, selection in enumerate(
                selections,
                start=1,
            ):
                source_path = selection["source_path"]
                suffix = source_path.suffix.lower()

                image_id = (
                    f"cifake_v3_{split}_{output_label}_"
                    f"{index:04d}"
                )

                destination_name = f"{image_id}{suffix}"

                destination_path = (
                    output_directories[output_label]
                    / destination_name
                )

                shutil.copy2(
                    source_path,
                    destination_path,
                )

                copied_hash = sha256_file(destination_path)

                if copied_hash != selection["source_sha256"]:
                    raise RuntimeError(
                        "Copied-file hash mismatch for "
                        f"{destination_path}"
                    )

                with Image.open(destination_path) as image:
                    image.load()

                    width, height = image.size
                    image_mode = image.mode
                    image_format = image.format or "UNKNOWN"

                source_project_relative_path = (
                    source_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                )

                selected_relative_path = (
                    destination_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                )

                rows.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "label": output_label,
                        "label_id": label_id,
                        "source_dataset": "CIFAKE",
                        "source_version": 3,
                        "source_original_split": (
                            expected_source_split
                        ),
                        "source_original_label": (
                            source_label
                        ),
                        "source_project_relative_path": (
                            source_project_relative_path
                        ),
                        "source_dataset_relative_path": (
                            selection[
                                "source_dataset_relative_path"
                            ]
                        ),
                        "selected_relative_path": (
                            selected_relative_path
                        ),
                        "source_sha256": (
                            selection["source_sha256"]
                        ),
                        "copied_sha256": copied_hash,
                        "selection_rank": (
                            selection["selection_rank"]
                        ),
                        "file_size_bytes": (
                            destination_path.stat().st_size
                        ),
                        "width": width,
                        "height": height,
                        "mode": image_mode,
                        "format": image_format,
                    }
                )

    rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            int(row["label_id"]),
            row["image_id"],
        )
    )

    manifest_path = (
        PROJECT_ROOT / output_config["manifest"]
    )

    write_manifest(
        manifest_path=manifest_path,
        rows=rows,
    )

    manifest_sha256 = sha256_file(manifest_path)

    count_summary = Counter(
        (row["split"], row["label"])
        for row in rows
    )

    total_bytes = sum(
        int(row["file_size_bytes"])
        for row in rows
    )

    selection_metadata = {
        "selection_name": selection_config["name"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "algorithm": selection_config["algorithm"],
        "reject_exact_duplicates": (
            selection_config["reject_exact_duplicates"]
        ),
        "preserve_original_test_boundary": (
            selection_config[
                "preserve_original_test_boundary"
            ]
        ),
        "source_dataset": "CIFAKE",
        "source_version": 3,
        "source_metadata": source_config["metadata"],
        "selection_manifest": output_config["manifest"],
        "selection_manifest_sha256": manifest_sha256,
        "selected_total": len(rows),
        "selected_unique_sha256": len(
            {row["source_sha256"] for row in rows}
        ),
        "duplicate_candidates_skipped": (
            duplicate_candidates_skipped
        ),
        "selected_bytes": total_bytes,
        "counts": {
            split: {
                label: count_summary[(split, label)]
                for label in ["real", "ai_generated"]
            }
            for split in [
                "train",
                "validation",
                "test",
            ]
        },
        "rules": rule_config,
    }

    metadata_path = (
        PROJECT_ROOT / output_config["metadata"]
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        json.dumps(
            selection_metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("========== PROTOTYPE SELECTION ==========")

    for split in ["train", "validation", "test"]:
        for label in ["real", "ai_generated"]:
            print(
                f"{split:10} {label:12} "
                f"{count_summary[(split, label)]:>4}"
            )

    print()
    print("Selected total:", len(rows))
    print(
        "Unique SHA-256 hashes:",
        len({row["source_sha256"] for row in rows}),
    )
    print(
        "Duplicate candidates skipped:",
        duplicate_candidates_skipped,
    )
    print(
        "Selected image bytes:",
        f"{total_bytes / (1024 ** 2):.2f} MiB",
    )
    print(
        "Manifest:",
        manifest_path.relative_to(PROJECT_ROOT),
    )
    print(
        "Metadata:",
        metadata_path.relative_to(PROJECT_ROOT),
    )
    print("Manifest SHA-256:", manifest_sha256)
    print()
    print("CIFAKE prototype selection completed.")

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
