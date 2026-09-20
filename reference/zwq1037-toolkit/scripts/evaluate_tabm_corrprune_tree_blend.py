"""Save the selected correlated-pruned TabM plus three-tree blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-BLEND-005"
SOURCE_FILES = {
    "EXP-TABM-001-MSE": "exp-tabm-001_valid.feather",
    "EXP-TABM-006-CORRPRUNE": "exp-tabm-006-corrprune_valid.feather",
    "EXP-TABM-003-COSINE": "exp-tabm-003-cosine_valid.feather",
    "EXP-TREE-053R-XGBOOST": "exp-tree-053r_valid.feather",
    "EXP-TREE-068-LIGHTGBM": "exp-tree-068_valid.feather",
    "EXP-TREE-007-HISTGB": "hist_gradient_boosting_target_centered_level2_valid.feather",
}
SELECTED_WEIGHTS = {
    "EXP-TABM-001-MSE": 0.35,
    "EXP-TABM-006-CORRPRUNE": 0.20,
    "EXP-TABM-003-COSINE": 0.15,
    "EXP-TREE-053R-XGBOOST": 0.10,
    "EXP-TREE-068-LIGHTGBM": 0.10,
    "EXP-TREE-007-HISTGB": 0.10,
}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["EXP-TABM-001-MSE"]
    validation_rows = reference[["sample_id", "month", "target"]].copy()
    normalized = {}
    source_norms = {}
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        prediction = frame["prediction"].to_numpy(dtype=np.float64)
        source_norms[name] = float(np.linalg.norm(prediction))
        normalized[name] = prediction / source_norms[name]
    selected_prediction = sum(
        SELECTED_WEIGHTS[name] * normalized[name] for name in SELECTED_WEIGHTS
    )

    baseline = pd.read_feather(prediction_dir / "exp-blend-003_valid.feather")
    assert_prediction_alignment(baseline, validation_rows)
    tabm_config = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-001"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "35% original MSE TabM, 20% correlation-pruned MSE TabM, 15% "
            "cosine-loss TabM, and 10% each XGBoost, LightGBM, HistGB after "
            "per-source L2 normalization"
        ),
        model=None,
        predictions=selected_prediction,
        validation_rows=validation_rows,
        baseline_predictions=baseline,
        feature_columns=list(tabm_config["feature_columns"]),
        parameters={
            "weights": SELECTED_WEIGHTS,
            "normalization": "each source validation vector divided by its global L2 norm",
            "selection_rule": (
                "largest coarse correlation-pruned TabM weight with overall at "
                "least submitted EXP-BLEND-001 and all primary robustness "
                "diagnostics improving versus EXP-BLEND-003; 25% failed the "
                "predeclared overall floor"
            ),
            "correlation_pruned_weight_grid": [
                0.0,
                0.025,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
            ],
        },
        training_seconds=0.0,
        baseline_id="EXP-BLEND-003",
        save_model=False,
    )
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    metadata.update(
        {
            "model_family": "normalized_prediction_blend",
            "source_models": list(SOURCE_FILES),
            "source_prediction_files": SOURCE_FILES,
            "source_prediction_norms": source_norms,
            "audit_grid_path": str(
                project_dir
                / "data"
                / "interim"
                / "tree_experiments"
                / "EXP-BLEND-005-AUDIT"
                / "corrprune_grid.csv"
            ),
            "overall_floor_source": "EXP-BLEND-001 overall=0.17571880874787868",
        }
    )
    metadata.pop("xgboost_version", None)
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
