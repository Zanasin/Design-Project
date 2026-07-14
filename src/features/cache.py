from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from src.data import (
    ImagePreprocessor,
    ImageRecord,
    ManifestImageDataset,
)
from src.features.handcrafted import (
    FEATURE_ORDER,
    HandcraftedFeatureConfig,
    extract_handcrafted_features,
)


CACHE_ARRAY_NAMES = (
    "hog",
    "lbp",
    "fft",
    "dct",
    "combined",
    "labels",
    "image_ids",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def save_npy_atomic(
    path: Path,
    array: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".temporary"
    )

    with temporary_path.open("wb") as file:
        np.save(
            file,
            array,
            allow_pickle=False,
        )

    temporary_path.replace(path)


def _array_metadata(
    path: Path,
    output_root: Path,
) -> dict[str, Any]:
    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    try:
        return {
            "path": path.relative_to(
                output_root
            ).as_posix(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    finally:
        del array


def build_split_feature_cache(
    *,
    records: Sequence[ImageRecord],
    project_root: Path,
    output_root: Path,
    preprocessor: ImagePreprocessor,
    feature_config: HandcraftedFeatureConfig,
    split: str,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    if not records:
        raise ValueError(
            "Cannot cache an empty record sequence"
        )

    if any(
        record.split != split
        for record in records
    ):
        raise ValueError(
            "All records must match the requested split"
        )

    split_directory = output_root / split

    if split_directory.exists():
        existing_files = list(
            split_directory.iterdir()
        )

        if existing_files and not force:
            raise FileExistsError(
                f"Feature cache already exists for "
                f"{split}. Use --force to rebuild it."
            )

        if force:
            shutil.rmtree(split_directory)

    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = ManifestImageDataset(
        records=records,
        project_root=project_root,
        preprocessor=preprocessor,
        output_mode="numpy",
    )

    record_count = len(dataset)
    dimensions = feature_config.dimensions

    matrices = {
        name: np.empty(
            (
                record_count,
                dimensions[name],
            ),
            dtype=np.float32,
        )
        for name in (
            *FEATURE_ORDER,
            "combined",
        )
    }

    labels = np.empty(
        record_count,
        dtype=np.int64,
    )

    image_ids: list[str] = []

    indices = range(record_count)

    if show_progress:
        indices = tqdm(
            indices,
            total=record_count,
            desc=f"Extracting {split}",
            unit="image",
        )

    for index in indices:
        sample = dataset[index]

        bundle = extract_handcrafted_features(
            sample["image"],
            feature_config,
        )

        for name, vector in bundle.as_dict().items():
            matrices[name][index] = vector

        labels[index] = int(sample["label"])
        image_ids.append(
            str(sample["image_id"])
        )

    image_id_array = np.asarray(
        image_ids,
        dtype=np.str_,
    )

    arrays = {
        **matrices,
        "labels": labels,
        "image_ids": image_id_array,
    }

    for name in CACHE_ARRAY_NAMES:
        save_npy_atomic(
            split_directory / f"{name}.npy",
            arrays[name],
        )

    files = {
        name: _array_metadata(
            split_directory / f"{name}.npy",
            output_root,
        )
        for name in CACHE_ARRAY_NAMES
    }

    class_counts = Counter(
        record.label
        for record in records
    )

    return {
        "record_count": record_count,
        "class_counts": {
            label: class_counts[label]
            for label in (
                "real",
                "ai_generated",
            )
        },
        "files": files,
    }


def write_json_atomic(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".temporary"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def verify_feature_cache_metadata(
    metadata_path: Path,
    project_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if metadata["cache_version"] != 1:
        raise ValueError(
            "Unsupported feature-cache version"
        )

    if expected_manifest_sha256 is not None:
        if (
            metadata["manifest_sha256"]
            != expected_manifest_sha256
        ):
            raise ValueError(
                "Feature cache uses a different manifest"
            )

    if expected_config_sha256 is not None:
        if (
            metadata["feature_config_sha256"]
            != expected_config_sha256
        ):
            raise ValueError(
                "Feature cache uses a different configuration"
            )

    resolved_project_root = project_root.resolve()

    cache_root = (
        resolved_project_root
        / metadata["output_root"]
    ).resolve()

    if not cache_root.is_relative_to(
        resolved_project_root
    ):
        raise ValueError(
            "Feature-cache root escapes project directory"
        )

    for split, split_metadata in metadata[
        "splits"
    ].items():
        record_count = int(
            split_metadata["record_count"]
        )

        for name, file_metadata in split_metadata[
            "files"
        ].items():
            relative_path = Path(
                file_metadata["path"]
            )

            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise ValueError(
                    f"Unsafe cache path for {split}/{name}"
                )

            path = (
                cache_root
                / relative_path
            ).resolve()

            if not path.is_relative_to(cache_root):
                raise ValueError(
                    f"Cache path escapes output root for "
                    f"{split}/{name}"
                )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {split}/{name} cache: "
                    f"{path}"
                )

            actual_hash = sha256_file(path)

            if actual_hash != file_metadata["sha256"]:
                raise ValueError(
                    f"Hash mismatch for {split}/{name}"
                )

            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            try:
                if list(array.shape) != file_metadata[
                    "shape"
                ]:
                    raise ValueError(
                        f"Shape mismatch for "
                        f"{split}/{name}"
                    )

                if str(array.dtype) != file_metadata[
                    "dtype"
                ]:
                    raise ValueError(
                        f"Dtype mismatch for "
                        f"{split}/{name}"
                    )

                if array.shape[0] != record_count:
                    raise ValueError(
                        f"Record-count mismatch for "
                        f"{split}/{name}"
                    )
            finally:
                del array

    return metadata
