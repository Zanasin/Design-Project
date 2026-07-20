from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "prototype_selection.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_rows() -> list[dict[str, str]]:
    config = load_config()

    manifest_path = (
        PROJECT_ROOT
        / config["outputs"]["manifest"]
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def test_selection_counts_and_boundaries() -> None:
    config = load_config()
    rows = load_rows()

    assert len(rows) == 1000

    expected_counts = {
        ("train", "real"): 350,
        ("train", "ai_generated"): 350,
        ("validation", "real"): 75,
        ("validation", "ai_generated"): 75,
        ("test", "real"): 75,
        ("test", "ai_generated"): 75,
    }

    actual_counts = Counter(
        (row["split"], row["label"])
        for row in rows
    )

    assert actual_counts == expected_counts

    for row in rows:
        if row["split"] in {"train", "validation"}:
            assert row["source_original_split"] == "train"
        elif row["split"] == "test":
            assert row["source_original_split"] == "test"
        else:
            raise AssertionError(
                f"Unexpected split: {row['split']}"
            )

        if row["label"] == "real":
            assert row["label_id"] == "0"
            assert row["source_original_label"] == "real"
        elif row["label"] == "ai_generated":
            assert row["label_id"] == "1"
            assert row["source_original_label"] == "fake"
        else:
            raise AssertionError(
                f"Unexpected label: {row['label']}"
            )


def test_selection_identifiers_and_hashes_are_unique() -> None:
    rows = load_rows()

    image_ids = [row["image_id"] for row in rows]
    source_paths = [
        row["source_project_relative_path"]
        for row in rows
    ]
    selected_paths = [
        row["selected_relative_path"]
        for row in rows
    ]
    content_hashes = [
        row["source_sha256"]
        for row in rows
    ]

    assert len(set(image_ids)) == 1000
    assert len(set(source_paths)) == 1000
    assert len(set(selected_paths)) == 1000
    assert len(set(content_hashes)) == 1000


def test_selected_images_exist_decode_and_match_hashes() -> None:
    rows = load_rows()

    for row in rows:
        source_path = (
            PROJECT_ROOT
            / row["source_project_relative_path"]
        )

        selected_path = (
            PROJECT_ROOT
            / row["selected_relative_path"]
        )

        assert source_path.is_file()
        assert selected_path.is_file()

        source_hash = sha256_file(source_path)
        selected_hash = sha256_file(selected_path)

        assert source_hash == row["source_sha256"]
        assert selected_hash == row["copied_sha256"]
        assert source_hash == selected_hash

        with Image.open(selected_path) as image:
            image.load()

            assert image.size == (
                int(row["width"]),
                int(row["height"]),
            )

            assert image.width > 0
            assert image.height > 0
            assert image.mode == row["mode"]


def test_selection_metadata_matches_manifest() -> None:
    config = load_config()
    rows = load_rows()

    manifest_path = (
        PROJECT_ROOT
        / config["outputs"]["manifest"]
    )

    metadata_path = (
        PROJECT_ROOT
        / config["outputs"]["metadata"]
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["source_dataset"] == "CIFAKE"
    assert metadata["source_version"] == 3
    assert metadata["seed"] == 42
    assert metadata["selected_total"] == 1000
    assert metadata["selected_unique_sha256"] == 1000
    assert metadata[
        "preserve_original_test_boundary"
    ] is True

    assert metadata[
        "selection_manifest_sha256"
    ] == sha256_file(manifest_path)

    expected_counts = {
        "train": {
            "real": 350,
            "ai_generated": 350,
        },
        "validation": {
            "real": 75,
            "ai_generated": 75,
        },
        "test": {
            "real": 75,
            "ai_generated": 75,
        },
    }

    assert metadata["counts"] == expected_counts
    assert len(rows) == metadata["selected_total"]
