from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation import (
    build_prototype_comparison,
    render_comparison_json,
    render_markdown_summary,
    render_metrics_csv,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prototype_comparison.yaml"
)


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}"
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


def write_atomic(
    *,
    path: Path,
    content: str,
) -> None:
    temporary = path.with_name(
        path.name + ".temporary"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if temporary.exists():
        temporary.unlink()

    try:
        temporary.write_text(
            content,
            encoding="utf-8",
        )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the unified comparison "
            "for all finalized prototype models."
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
        help=(
            "Replace existing comparison outputs."
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify existing outputs against the "
            "finalized source artifacts."
        ),
    )

    args = parser.parse_args()

    if args.force and args.verify_only:
        raise ValueError(
            "--force and --verify-only "
            "cannot be combined"
        )

    config_path = args.config.resolve()
    config_file = load_yaml(config_path)

    config = config_file[
        "prototype_comparison"
    ]

    output_paths = {
        name: resolve_project_path(path)
        for name, path
        in config["outputs"].items()
    }

    comparison = build_prototype_comparison(
        config_file=config_file,
        project_root=PROJECT_ROOT,
    )

    expected_content = {
        "comparison_report": (
            render_comparison_json(
                comparison
            )
        ),
        "metrics_table": (
            render_metrics_csv(
                comparison
            )
        ),
        "markdown_summary": (
            render_markdown_summary(
                comparison
            )
        ),
    }

    if args.verify_only:
        for name, path in (
            output_paths.items()
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Comparison output missing: {path}"
                )

            actual = path.read_text(
                encoding="utf-8"
            )

            if actual != expected_content[name]:
                raise ValueError(
                    f"Comparison output is stale or "
                    f"changed: {path}"
                )

        print(
            "Prototype comparison verification passed."
        )
        print(
            "Models:",
            len(comparison["models"]),
        )
        print(
            "Test records:",
            comparison["scope"][
                "test_records"
            ],
        )
        print(
            "Best F1:",
            comparison["summary"][
                "best_overall_by_f1"
            ],
        )
        print(
            "Best recall:",
            comparison["summary"][
                "best_recall"
            ],
        )
        print(
            "All models wrong:",
            comparison["agreement"][
                "all_three_wrong"
            ],
        )

        return 0

    existing = [
        path
        for path in output_paths.values()
        if path.exists()
    ]

    if existing and not args.force:
        raise FileExistsError(
            "Prototype comparison outputs already "
            "exist. Use --verify-only or --force: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    for name, path in (
        output_paths.items()
    ):
        write_atomic(
            path=path,
            content=expected_content[name],
        )

    print("========== PROTOTYPE COMPARISON ==========")

    for model in comparison["models"]:
        metrics = model["metrics"]

        print(
            model["display_name"],
            "| accuracy:",
            f"{metrics['accuracy']:.4f}",
            "| F1:",
            f"{metrics['f1']:.4f}",
            "| ROC-AUC:",
            f"{metrics['roc_auc']:.4f}",
        )

    print()
    print(
        "Best overall by F1:",
        comparison["summary"][
            "best_overall_by_f1"
        ],
    )
    print(
        "Best recall:",
        comparison["summary"][
            "best_recall"
        ],
    )
    print(
        "Smallest artifact:",
        comparison["summary"][
            "smallest_artifact"
        ],
    )
    print(
        "All three wrong:",
        comparison["agreement"][
            "all_three_wrong"
        ],
    )
    print()
    print(
        "Comparison report:",
        output_paths[
            "comparison_report"
        ].relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Metrics table:",
        output_paths[
            "metrics_table"
        ].relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print(
        "Markdown summary:",
        output_paths[
            "markdown_summary"
        ].relative_to(
            PROJECT_ROOT.resolve()
        ),
    )
    print()
    print(
        "PASS: all models use identical test "
        "IDs and labels."
    )
    print(
        "PASS: all recorded metrics match "
        "their prediction files."
    )
    print(
        "PASS: unified prototype comparison "
        "was generated."
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
