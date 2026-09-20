"""Run TabM after training-only pruning of near-duplicate EXP053R features."""

from __future__ import annotations

import json
from pathlib import Path

import exp_tabm_001_exp053r_features as experiment


EXPERIMENT_ID = "EXP-TABM-006-CORRPRUNE"
EXPECTED_FEATURE_COUNT = 293
BASE_LOADER = experiment.load_exp053r_data


def load_pruned_data(project_dir: Path):
    """Keep the training-only correlation-pruned feature subset."""
    model_data, feature_columns, public_features, dropped_public_features = (
        BASE_LOADER(project_dir)
    )
    selection_path = (
        project_dir / "data" / "interim" / "exp053r_correlated_feature_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_source"] != "training months 0-59 only":
        raise ValueError("Feature selection must use training months only.")
    kept = set(selection["kept_features_in_gain_order"])

    # 保留原始列顺序，只删除训练期内相关系数绝对值超过 0.995 的冗余列。
    # Preserve source order; remove only redundancies found above 0.995 in training.
    pruned_features = [name for name in feature_columns if name in kept]
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
        project_dir / "data" / "interim" / "exp053r_correlated_feature_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "description": (
                "official TabM on 293 EXP053R features after training-only "
                "absolute-Pearson correlation pruning at 0.995"
            ),
            "feature_selection": {
                "method": "training-only absolute Pearson correlation pruning",
                "threshold": selection["correlation_threshold"],
                "selection_source": selection["selection_source"],
                "original_feature_count": selection["original_feature_count"],
                "kept_feature_count": selection["kept_feature_count"],
                "removed_feature_count": selection["removed_feature_count"],
                "selection_file": str(selection_path.relative_to(project_dir)),
            },
        }
    )
    config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
