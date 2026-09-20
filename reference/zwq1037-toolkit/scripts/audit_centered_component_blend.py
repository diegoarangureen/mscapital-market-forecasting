"""Audit label-free component centering before TabM/XGBoost blending."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


TABM_WEIGHT_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    tabm = pd.read_feather(prediction_dir / "exp-tabm-001_valid.feather")
    tree = pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather")
    validation_rows = tabm[["sample_id", "month", "target"]]
    assert_prediction_alignment(tree, validation_rows)
    target = tabm["target"].to_numpy(dtype=np.float64)
    months = tabm["month"].to_numpy()
    month_values = np.sort(np.unique(months))

    raw_tabm = tabm["prediction"].to_numpy(dtype=np.float64)
    raw_tree = tree["prediction"].to_numpy(dtype=np.float64)
    centered_tabm = raw_tabm - raw_tabm.mean()
    centered_tree = raw_tree - raw_tree.mean()
    centered_tabm /= np.linalg.norm(centered_tabm)
    centered_tree /= np.linalg.norm(centered_tree)
    primary_mask = (months >= 62) & (months != 66)

    rows = []
    for tabm_weight in TABM_WEIGHT_GRID:
        tree_weight = 1.0 - tabm_weight
        prediction = tabm_weight * centered_tabm + tree_weight * centered_tree
        monthly_scores = np.asarray(
            [
                cosine(target[months == month], prediction[months == month])
                for month in month_values
            ]
        )
        primary_scores = monthly_scores[
            np.asarray([(month >= 62 and month != 66) for month in month_values])
        ]
        rows.append(
            {
                "tabm_weight": tabm_weight,
                "xgboost_weight": tree_weight,
                "overall_cosine": cosine(target, prediction),
                "cosine_62_70_without_66": cosine(
                    target[primary_mask], prediction[primary_mask]
                ),
                "cosine_67_70": cosine(
                    target[months >= 67], prediction[months >= 67]
                ),
                "cosine_60_64": cosine(
                    target[months <= 64], prediction[months <= 64]
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
        / "EXP-POST-002-COMPONENT-CENTERING"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(run_dir / "centered_component_weight_grid.csv", index=False)
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "method": "center each source, L2 normalize each source, then blend",
                "uses_labels_for_transformation": False,
                "tabm_weight_grid": TABM_WEIGHT_GRID,
                "submission_policy": "audit only; preserve final submission until user chooses",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
