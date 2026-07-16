from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_HANDLE = (
    "birdy654/"
    "cifake-real-and-ai-generated-synthetic-images/"
    "versions/3"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "data" / "interim" / "cifake_source_v3"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "prototype"
    / "cifake_source.json"
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_COUNTS = {
    "train_real": 50_000,
    "train_fake": 50_000,
    "test_real": 10_000,
    "test_fake": 10_000,
}


def classify_image_path(path: Path) -> tuple[str | None, str | None]:
    """Infer the original CIFAKE split and label from directory names."""

    parts = {part.lower() for part in path.parts}

    split: str | None = None
    label: str | None = None

    if "train" in parts:
        split = "train"
    elif "test" in parts:
        split = "test"

    if "real" in parts:
        label = "real"
    elif "fake" in parts:
        label = "fake"

    return split, label


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Dataset handle:", DATASET_HANDLE)
    print(
        "Output directory:",
        OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT),
    )
    print()

    downloaded_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(OUTPUT_DIRECTORY),
        )
    ).resolve()

    print()
    print("KaggleHub returned:", downloaded_path)
    print("Scanning downloaded files...")

    image_files = sorted(
        path
        for path in downloaded_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    counts: Counter[str] = Counter()
    unclassified_files: list[str] = []

    for image_path in image_files:
        split, label = classify_image_path(image_path)

        if split is None or label is None:
            unclassified_files.append(
                image_path.relative_to(PROJECT_ROOT).as_posix()
            )
            continue

        counts[f"{split}_{label}"] += 1

    total_classified = sum(counts.values())
    total_bytes = sum(path.stat().st_size for path in image_files)

    print()
    print("========== CIFAKE SOURCE AUDIT ==========")

    for key in EXPECTED_COUNTS:
        print(f"{key:16} {counts[key]:>8,}")

    print(f"{'classified_total':16} {total_classified:>8,}")
    print(f"{'image_files':16} {len(image_files):>8,}")
    print(
        f"{'image_size':16} "
        f"{total_bytes / (1024 ** 2):>7.2f} MiB"
    )
    print(f"{'unclassified':16} {len(unclassified_files):>8,}")

    errors: list[str] = []

    for key, expected_count in EXPECTED_COUNTS.items():
        actual_count = counts[key]

        if actual_count != expected_count:
            errors.append(
                f"{key}: expected {expected_count}, found {actual_count}"
            )

    expected_total = sum(EXPECTED_COUNTS.values())

    if total_classified != expected_total:
        errors.append(
            "classified total: "
            f"expected {expected_total}, found {total_classified}"
        )

    if unclassified_files:
        errors.append(
            f"{len(unclassified_files)} image files could not be classified"
        )

    relative_download_root = downloaded_path.relative_to(
        PROJECT_ROOT
    ).as_posix()

    metadata = {
        "dataset_name": "CIFAKE",
        "dataset_handle": DATASET_HANDLE,
        "dataset_version": 3,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "download_root": relative_download_root,
        "kagglehub_version": version("kagglehub"),
        "image_suffixes": sorted(IMAGE_SUFFIXES),
        "counts": {
            key: counts[key]
            for key in EXPECTED_COUNTS
        },
        "classified_total": total_classified,
        "image_file_total": len(image_files),
        "image_bytes": total_bytes,
        "unclassified_image_count": len(unclassified_files),
        "audit_passed": not errors,
        "audit_errors": errors,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Metadata saved to:",
        METADATA_PATH.relative_to(PROJECT_ROOT),
    )

    if errors:
        print()
        print("AUDIT FAILED:")

        for error in errors:
            print("-", error)

        if unclassified_files:
            print()
            print("First unclassified files:")

            for file_path in unclassified_files[:10]:
                print("-", file_path)

        return 1

    print()
    print("CIFAKE source audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
