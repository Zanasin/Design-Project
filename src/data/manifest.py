from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast


SplitName = Literal["train", "validation", "test"]
LabelName = Literal["real", "ai_generated"]

ALLOWED_SPLITS = {
    "train",
    "validation",
    "test",
}

ALLOWED_LABELS = {
    "real",
    "ai_generated",
}

LABEL_TO_ID = {
    "real": 0,
    "ai_generated": 1,
}

REQUIRED_COLUMNS = {
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
}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    split: SplitName
    label: LabelName
    label_id: int
    source_dataset: str
    source_version: int
    source_original_split: str
    source_original_label: str
    source_project_relative_path: str
    source_dataset_relative_path: str
    selected_relative_path: str
    source_sha256: str
    copied_sha256: str
    selection_rank: str
    file_size_bytes: int
    width: int
    height: int
    mode: str
    image_format: str

    @classmethod
    def from_csv_row(
        cls,
        row: dict[str, str],
    ) -> ImageRecord:
        split = parse_split(row["split"])
        label = parse_label(row["label"])

        record = cls(
            image_id=row["image_id"],
            split=split,
            label=label,
            label_id=int(row["label_id"]),
            source_dataset=row["source_dataset"],
            source_version=int(row["source_version"]),
            source_original_split=row["source_original_split"],
            source_original_label=row["source_original_label"],
            source_project_relative_path=(
                row["source_project_relative_path"]
            ),
            source_dataset_relative_path=(
                row["source_dataset_relative_path"]
            ),
            selected_relative_path=(
                row["selected_relative_path"]
            ),
            source_sha256=row["source_sha256"],
            copied_sha256=row["copied_sha256"],
            selection_rank=row["selection_rank"],
            file_size_bytes=int(row["file_size_bytes"]),
            width=int(row["width"]),
            height=int(row["height"]),
            mode=row["mode"],
            image_format=row["format"],
        )

        validate_record(record)
        return record

    def absolute_path(
        self,
        project_root: Path,
    ) -> Path:
        resolved_root = project_root.resolve()
        resolved_path = (
            resolved_root / self.selected_relative_path
        ).resolve()

        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(
                "Manifest path escapes project root: "
                f"{self.selected_relative_path}"
            )

        return resolved_path


def parse_split(value: str) -> SplitName:
    if value not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split: {value!r}")

    return cast(SplitName, value)


def parse_label(value: str) -> LabelName:
    if value not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported label: {value!r}")

    return cast(LabelName, value)


def validate_sha256(
    value: str,
    field_name: str,
) -> None:
    if len(value) != 64:
        raise ValueError(
            f"{field_name} must contain 64 hexadecimal characters"
        )

    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} is not a valid SHA-256 digest"
        ) from exc


def validate_record(record: ImageRecord) -> None:
    if not record.image_id:
        raise ValueError("image_id cannot be empty")

    expected_label_id = LABEL_TO_ID[record.label]

    if record.label_id != expected_label_id:
        raise ValueError(
            f"Incorrect label_id for {record.image_id}: "
            f"expected {expected_label_id}, "
            f"found {record.label_id}"
        )

    if record.width <= 0 or record.height <= 0:
        raise ValueError(
            f"Invalid dimensions for {record.image_id}"
        )

    if record.file_size_bytes <= 0:
        raise ValueError(
            f"Invalid file size for {record.image_id}"
        )

    if not record.selected_relative_path:
        raise ValueError(
            f"Missing selected path for {record.image_id}"
        )

    validate_sha256(
        record.source_sha256,
        "source_sha256",
    )

    validate_sha256(
        record.copied_sha256,
        "copied_sha256",
    )

    if record.source_sha256 != record.copied_sha256:
        raise ValueError(
            f"Source and copied hashes differ for "
            f"{record.image_id}"
        )

    if record.split in {"train", "validation"}:
        expected_source_split = "train"
    else:
        expected_source_split = "test"

    if (
        record.source_original_split
        != expected_source_split
    ):
        raise ValueError(
            f"Source boundary violation for "
            f"{record.image_id}"
        )

    expected_source_label = (
        "real"
        if record.label == "real"
        else "fake"
    )

    if (
        record.source_original_label
        != expected_source_label
    ):
        raise ValueError(
            f"Source label mismatch for "
            f"{record.image_id}"
        )


def validate_uniqueness(
    records: Sequence[ImageRecord],
) -> None:
    fields = {
        "image_id": [
            record.image_id
            for record in records
        ],
        "selected_relative_path": [
            record.selected_relative_path
            for record in records
        ],
        "source_sha256": [
            record.source_sha256
            for record in records
        ],
    }

    for field_name, values in fields.items():
        if len(values) != len(set(values)):
            raise ValueError(
                f"Manifest contains duplicate {field_name} values"
            )


def load_manifest(
    manifest_path: Path,
) -> tuple[ImageRecord, ...]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        missing_columns = REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            raise ValueError(
                "Manifest is missing columns: "
                f"{sorted(missing_columns)}"
            )

        records: list[ImageRecord] = []

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                record = ImageRecord.from_csv_row(row)
            except Exception as exc:
                raise ValueError(
                    f"Invalid manifest row at line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError("Manifest contains no records")

    validate_uniqueness(records)
    return tuple(records)


def records_for_split(
    records: Sequence[ImageRecord],
    split: str,
) -> tuple[ImageRecord, ...]:
    split_name = parse_split(split)

    selected = tuple(
        record
        for record in records
        if record.split == split_name
    )

    if not selected:
        raise ValueError(
            f"No records found for split: {split_name}"
        )

    return selected


def development_records(
    records: Sequence[ImageRecord],
    split: str,
) -> tuple[ImageRecord, ...]:
    split_name = parse_split(split)

    if split_name == "test":
        raise ValueError(
            "The test split is evaluation-only and cannot "
            "be loaded through the development interface."
        )

    return records_for_split(
        records=records,
        split=split_name,
    )


def count_by_split_and_label(
    records: Sequence[ImageRecord],
) -> Counter[tuple[str, str]]:
    return Counter(
        (record.split, record.label)
        for record in records
    )
