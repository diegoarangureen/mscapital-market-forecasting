"""Evaluate a dual-loss TabM plus three-tree robustness blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-BLEND-003"
SOURCE_FILES = {
    "EXP-TABM-001-MSE": "exp-tabm-001_valid.feather",
    "EXP-TABM-003-COSINE": "exp-tabm-003-cosine_valid.feather",
    "EXP-TREE-053R-XGBOOST": "exp-tree-053r_valid.feather",
    "EXP-TREE-068-LIGHTGBM": "exp-tree-068_valid.feather",
    "EXP-TREE-007-HISTGB": "hist_gradient_boosting_target_centered_level2_valid.feather",
}
SELECTED_WEIGHTS = {
    "EXP-TABM-001-MSE": 0.55,
    "EXP-TABM-003-COSINE": 0.15,
    "EXP-TREE-053R-XGBOOST": 0.10,
    "EXP-TREE-068-LIGHTGBM": 0.10,
    "EXP-TREE-007-HISTGB": 0.10,
}
COSINE_TABM_GRID = [0.00, 0.05, 0.10, 0.15, 0.20]


def cosine_score(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction))
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["EXP-TABM-001-MSE"]
    validation_rows = reference[["sample_id", "month", "target"]].copy()
    for frame in frames.values():
        assert_prediction_alignment(frame, validation_rows)

    normalized = {}
    source_norms = {}
    for name, frame in frames.items():
        prediction = frame["prediction"].to_numpy(dtype=np.float64)
        norm = float(np.linalg.norm(prediction))
        normalized[name] = prediction / norm
        source_norms[name] = norm

    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    month_values = np.sort(np.unique(months))
    rows = []
    for cosine_tabm_weight in COSINE_TABM_GRID:
        mse_tabm_weight = 0.70 - cosine_tabm_weight
        prediction = (
            mse_tabm_weight * normalized["EXP-TABM-001-MSE"]
            + cosine_tabm_weight * normalized["EXP-TABM-003-COSINE"]
            + 0.10 * normalized["EXP-TREE-053R-XGBOOST"]
            + 0.10 * normalized["EXP-TREE-068-LIGHTGBM"]
            + 0.10 * normalized["EXP-TREE-007-HISTGB"]
        )
        monthly_scores = np.asarray(
            [
                cosine_score(target[months == month], prediction[months == month])
                for month in month_values
            ]
        )
        primary = monthly_scores[(month_values >= 62) & (month_values != 66)]
        rows.append(
            {
                "mse_tabm_weight": mse_tabm_weight,
                "cosine_tabm_weight": cosine_tabm_weight,
                "each_tree_weight": 0.10,
                "overall_cosine": cosine_score(target, prediction),
                "cosine_62_70_without_66": cosine_score(
                    target[(months >= 62) & (months != 66)],
                    prediction[(months >= 62) & (months != 66)],
                ),
                "cosine_67_70": cosine_score(
                    target[months >= 67], prediction[months >= 67]
                ),
                "cosine_60_64": cosine_score(
                    target[months <= 64], prediction[months <= 64]
                ),
                "primary_monthly_std": float(primary.std(ddof=0)),
                "primary_monthly_worst": float(primary.min()),
                "primary_monthly_q25": float(np.quantile(primary, 0.25)),
            }
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(run_dir / "blend_grid.csv", index=False)

    selected_prediction = sum(
        SELECTED_WEIGHTS[name] * normalized[name] for name in SELECTED_WEIGHTS
    )
    baseline = pd.read_feather(prediction_dir / "exp-blend-002_valid.feather")
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
    feature_columns = list(tabm_config["feature_columns"])
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "55% MSE TabM, 15% cosine-loss TabM, and 10% each XGBoost, "
            "LightGBM, HistGradientBoosting after per-model L2 normalization"
        ),
        model=None,
        predictions=selected_prediction,
        validation_rows=validation_rows,
        baseline_predictions=baseline,
        feature_columns=feature_columns,
        parameters={
            "weights": SELECTED_WEIGHTS,
            "normalization": "each source validation vector divided by its global L2 norm",
            "selection_rule": (
                "largest coarse cosine-TabM weight whose overall score remains "
                "above submitted EXP-BLEND-001"
            ),
            "cosine_tabm_grid": COSINE_TABM_GRID,
        },
        training_seconds=0.0,
        baseline_id="EXP-BLEND-002",
        save_model=False,
    )
    metadata.update(
        {
            "model_family": "normalized_prediction_blend",
            "training_device": "not_applicable",
            "source_models": list(SOURCE_FILES),
            "source_prediction_files": SOURCE_FILES,
            "source_prediction_norms": source_norms,
            "blend_grid_path": str(run_dir / "blend_grid.csv"),
            "overall_floor_source": "EXP-BLEND-001 overall=0.17571880874787868",
        }
    )
    metadata.pop("xgboost_version", None)
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(grid.to_string(index=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
