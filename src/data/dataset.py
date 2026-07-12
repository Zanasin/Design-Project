from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence, cast

from torch.utils.data import Dataset

from src.data.manifest import (
    ImageRecord,
    development_records,
    load_manifest,
    records_for_split,
)
from src.data.preprocessing import ImagePreprocessor


OutputMode = Literal[
    "pil",
    "numpy",
    "tensor",
]

ALLOWED_OUTPUT_MODES = {
    "pil",
    "numpy",
    "tensor",
}


class ManifestImageDataset(
    Dataset[dict[str, Any]]
):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        project_root: Path,
        preprocessor: ImagePreprocessor,
        *,
        output_mode: OutputMode = "tensor",
        verify_paths: bool = True,
    ) -> None:
        if not records:
            raise ValueError(
                "Dataset requires at least one record"
            )

        if output_mode not in ALLOWED_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported output mode: {output_mode}"
            )

        self.records = tuple(records)
        self.project_root = project_root.resolve()
        self.preprocessor = preprocessor
        self.output_mode = cast(
            OutputMode,
            output_mode,
        )

        if verify_paths:
            missing_paths = [
                record.selected_relative_path
                for record in self.records
                if not record.absolute_path(
                    self.project_root
                ).is_file()
            ]

            if missing_paths:
                examples = ", ".join(
                    missing_paths[:5]
                )

                raise FileNotFoundError(
                    f"{len(missing_paths)} dataset images "
                    f"are missing. Examples: {examples}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        record = self.records[index]

        image_path = record.absolute_path(
            self.project_root
        )

        if self.output_mode == "pil":
            image = self.preprocessor.load(
                image_path
            )

        elif self.output_mode == "numpy":
            image = self.preprocessor.to_numpy(
                image_path,
                scale_to_unit_interval=True,
            )

        else:
            image = self.preprocessor.to_tensor(
                image_path,
                normalise_for_imagenet=True,
            )

        return {
            "image": image,
            "label": record.label_id,
            "image_id": record.image_id,
            "split": record.split,
            "class_name": record.label,
            "path": record.selected_relative_path,
        }


def build_development_dataset(
    manifest_path: Path,
    project_root: Path,
    preprocessor: ImagePreprocessor,
    *,
    split: str,
    output_mode: OutputMode = "tensor",
) -> ManifestImageDataset:
    records = load_manifest(manifest_path)

    selected_records = development_records(
        records=records,
        split=split,
    )

    return ManifestImageDataset(
        records=selected_records,
        project_root=project_root,
        preprocessor=preprocessor,
        output_mode=output_mode,
    )


def build_evaluation_dataset(
    manifest_path: Path,
    project_root: Path,
    preprocessor: ImagePreprocessor,
    *,
    split: str = "test",
    output_mode: OutputMode = "tensor",
) -> ManifestImageDataset:
    records = load_manifest(manifest_path)

    selected_records = records_for_split(
        records=records,
        split=split,
    )

    return ManifestImageDataset(
        records=selected_records,
        project_root=project_root,
        preprocessor=preprocessor,
        output_mode=output_mode,
    )
