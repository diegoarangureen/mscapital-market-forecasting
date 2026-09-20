"""Save a robustness-first TabM + EXP053R tree blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save
from exp_tree_052_053_xgboost_gpu_last15 import finalize_metadata
from train_full_xgboost_exp051_submission import selected_public_features


EXPERIMENT_ID = "EXP-BLEND-001"
TABM_WEIGHT = 0.75
GRID_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


def cosine_score(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.dot(target, prediction))
    denominator = float(np.linalg.norm(target) * np.linalg.norm(prediction))
    return numerator / denominator


def subset_score(
    target: np.ndarray,
    prediction: np.ndarray,
    months: np.ndarray,
    mask: np.ndarray,
) -> float:
    return cosine_score(target[mask], prediction[mask])


def normalized_prediction(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    norm = float(np.linalg.norm(prediction))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Prediction norm must be finite and non-zero.")
    return prediction / norm, norm


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    tabm = pd.read_feather(prediction_dir / "exp-tabm-001_valid.feather")
    tree = pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather")
    assert_prediction_alignment(tabm, tree[["sample_id", "month", "target"]])

    tabm_normalized, tabm_norm = normalized_prediction(tabm)
    tree_normalized, tree_norm = normalized_prediction(tree)
    target = tabm["target"].to_numpy(dtype=np.float64)
    months = tabm["month"].to_numpy()
    month_values = np.sort(np.unique(months))

    rows = []
    for tabm_weight in GRID_WEIGHTS:
        prediction = (
            tabm_weight * tabm_normalized
            + (1.0 - tabm_weight) * tree_normalized
        )
        monthly_scores = np.asarray(
            [
                subset_score(target, prediction, months, months == month)
                for month in month_values
            ]
        )
        primary_month_mask = (month_values >= 62) & (month_values != 66)
        primary_scores = monthly_scores[primary_month_mask]
        rows.append(
            {
                "tabm_weight": tabm_weight,
                "tree_weight": 1.0 - tabm_weight,
                "overall_cosine": cosine_score(target, prediction),
                "cosine_62_70_without_66": subset_score(
                    target,
                    prediction,
                    months,
                    (months >= 62) & (months != 66),
                ),
                "cosine_67_70": subset_score(
                    target, prediction, months, months >= 67
                ),
                "cosine_60_64": subset_score(
                    target, prediction, months, months <= 64
                ),
                "primary_monthly_std": float(primary_scores.std(ddof=0)),
                "primary_monthly_worst": float(primary_scores.min()),
                "primary_monthly_q25": float(np.quantile(primary_scores, 0.25)),
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

    selected_prediction = (
        TABM_WEIGHT * tabm_normalized
        + (1.0 - TABM_WEIGHT) * tree_normalized
    )
    validation_rows = tabm[["sample_id", "month", "target"]].copy()
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
    parameters = {
        "tabm_weight": TABM_WEIGHT,
        "tree_weight": 1.0 - TABM_WEIGHT,
        "normalization": "per-model global L2 norm",
        "selection_rule": "maximize robustness-first metrics on a coarse fixed grid",
        "grid_weights": GRID_WEIGHTS,
    }
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "Robustness-first normalized blend of 75% EXP-TABM-001 and 25% "
            "EXP-TREE-053R; selected from a coarse predeclared weight grid"
        ),
        model=None,
        predictions=selected_prediction,
        validation_rows=validation_rows,
        baseline_predictions=tabm,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=0.0,
        baseline_id="EXP-TABM-001",
        save_model=False,
    )
    public_features, dropped_public_features = selected_public_features(project_dir)
    finalize_metadata(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        metadata=metadata,
        baseline_predictions=tabm,
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )
    metadata.update(
        {
            "model_family": "normalized_prediction_blend",
            "training_device": "not_applicable",
            "source_models": ["EXP-TABM-001", "EXP-TREE-053R"],
            "source_prediction_norms": {
                "EXP-TABM-001": tabm_norm,
                "EXP-TREE-053R": tree_norm,
            },
            "blend_grid_path": str(run_dir / "blend_grid.csv"),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(grid.to_string(index=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
