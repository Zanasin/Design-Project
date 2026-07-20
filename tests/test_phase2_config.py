from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "prototype.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_prototype_counts_and_balance() -> None:
    config = load_config()
    dataset = config["dataset"]

    assert dataset["total_images"] == 1000
    assert dataset["classes"]["real"] == 500
    assert dataset["classes"]["ai_generated"] == 500

    splits = dataset["splits"]

    assert sum(split["total"] for split in splits.values()) == 1000
    assert sum(split["real"] for split in splits.values()) == 500
    assert sum(split["ai_generated"] for split in splits.values()) == 500

    for split in splits.values():
        assert split["real"] == split["ai_generated"]
        assert split["total"] == split["real"] + split["ai_generated"]


def test_prototype_safety_rules() -> None:
    config = load_config()

    assert config["prototype"]["purpose"] == "pipeline_validation_only"
    assert config["prototype"]["final_benchmark"] is False

    assert config["models"]["resnet18"]["pretrained"] is True
    assert config["models"]["resnet18"]["train_from_scratch"] is False
    assert config["models"]["resnet18"]["freeze_backbone"] is True

    rules = config["rules"]

    assert rules["balanced_classes"] is True
    assert rules["stratified_splits"] is True
    assert rules["fit_preprocessing_on_training_only"] is True
    assert rules["use_test_for_model_selection"] is False
    assert rules["no_final_research_claims"] is True
    assert rules["do_not_merge_with_final_dataset_without_reaudit"] is True


def test_prototype_directories_exist() -> None:
    required_directories = [
        "data/raw/prototype/real",
        "data/raw/prototype/ai_generated",
        "data/interim/prototype",
        "data/manifests/prototype",
        "features/prototype",
        "models/prototype/svm",
        "models/prototype/xgboost",
        "models/prototype/resnet18",
        "results/tables/prototype",
        "results/plots/prototype",
    ]

    for relative_path in required_directories:
        path = PROJECT_ROOT / relative_path
        assert path.is_dir(), f"Missing directory: {relative_path}"
