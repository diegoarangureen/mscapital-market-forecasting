"""Run TabM after dropping the lowest training-only XGBoost-gain features."""

from __future__ import annotations

import json
from pathlib import Path

import exp_tabm_001_exp053r_features as experiment


EXPERIMENT_ID = "EXP-TABM-007-GAINPRUNE"
EXPECTED_FEATURE_COUNT = 277
BASE_LOADER = experiment.load_exp053r_data


def load_pruned_data(project_dir: Path):
    """Remove the bottom 10% features selected using months 0--59 only."""
    model_data, feature_columns, public_features, dropped_public_features = (
        BASE_LOADER(project_dir)
    )
    selection_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-064"
        / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_source"] != "EXP-TREE-053R training-only total_gain":
        raise ValueError("Gain pruning must come from training-only feature importance.")
    removed = set(selection["removed_features"])

    # 只删除训练期总 gain 最低的 10%，保持其余列的原始顺序。
    # Drop only the bottom training-gain decile and preserve source column order.
    pruned_features = [name for name in feature_columns if name not in removed]
    if len(pruned_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} pruned features, got "
            f"{len(pruned_features)}."
        )
    return model_data, pruned_features, public_features, dropped_public_features


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.load_exp053r_data = load_pruned_data
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    config_path = run_dir / "config.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    selection_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-064"
        / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "description": (
                "official TabM on 277 EXP053R features after dropping the bottom "
                "10% by training-only XGBoost total_gain"
            ),
            "feature_selection": {
                "method": "drop bottom 10% by training-only XGBoost total_gain",
                "selection_source": selection["selection_source"],
                "drop_fraction": selection["drop_fraction"],
                "removed_feature_count": selection["drop_count"],
                "kept_feature_count": selection["kept_count"],
                "removed_total_gain_share": selection["removed_total_gain_share"],
                "selection_file": str(selection_path.relative_to(project_dir)),
            },
        }
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
