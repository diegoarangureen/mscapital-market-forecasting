"""Audit small gain-pruned TabM weights inside EXP-BLEND-005."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


SOURCE_FILES = {
    "tabm_mse": "exp-tabm-001_valid.feather",
    "tabm_corrprune": "exp-tabm-006-corrprune_valid.feather",
    "tabm_gainprune": "exp-tabm-007-gainprune_valid.feather",
    "tabm_cosine": "exp-tabm-003-cosine_valid.feather",
    "xgboost": "exp-tree-053r_valid.feather",
    "lightgbm": "exp-tree-068_valid.feather",
    "histgb": "hist_gradient_boosting_target_centered_level2_valid.feather",
}
GAIN_PRUNED_WEIGHT_GRID = [0.0, 0.025, 0.05, 0.10, 0.15]


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_mse"]
    validation_rows = reference[["sample_id", "month", "target"]]
    normalized = {}
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        values = frame["prediction"].to_numpy(dtype=np.float64)
        normalized[name] = values / np.linalg.norm(values)
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    month_values = np.sort(np.unique(months))
    primary_mask = (months >= 62) & (months != 66)
    baseline_prediction = (
        0.35 * normalized["tabm_mse"]
        + 0.20 * normalized["tabm_corrprune"]
        + 0.15 * normalized["tabm_cosine"]
        + 0.10 * normalized["xgboost"]
        + 0.10 * normalized["lightgbm"]
        + 0.10 * normalized["histgb"]
    )

    rows = []
    for gain_weight in GAIN_PRUNED_WEIGHT_GRID:
        prediction = (
            (0.35 - gain_weight) * normalized["tabm_mse"]
            + 0.20 * normalized["tabm_corrprune"]
            + gain_weight * normalized["tabm_gainprune"]
            + 0.15 * normalized["tabm_cosine"]
            + 0.10 * normalized["xgboost"]
            + 0.10 * normalized["lightgbm"]
            + 0.10 * normalized["histgb"]
        )
        monthly_scores = np.asarray(
            [
                cosine(target[months == month], prediction[months == month])
                for month in month_values
            ]
        )
        primary_scores = monthly_scores[
            np.asarray([(month >= 62 and month != 66) for month in month_values])
        ]
        lomo_changes = []
        for omitted_month in month_values[(month_values >= 62) & (month_values != 66)]:
            mask = primary_mask & (months != omitted_month)
            lomo_changes.append(
                cosine(target[mask], prediction[mask])
                - cosine(target[mask], baseline_prediction[mask])
            )
        rows.append(
            {
                "tabm_mse_weight": 0.35 - gain_weight,
                "tabm_corrprune_weight": 0.20,
                "tabm_gainprune_weight": gain_weight,
                "tabm_cosine_weight": 0.15,
                "each_tree_weight": 0.10,
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
                "primary_lomo_min_change_vs_blend005": float(np.min(lomo_changes)),
                "primary_lomo_mean_change_vs_blend005": float(np.mean(lomo_changes)),
            }
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-006-AUDIT"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(run_dir / "gainprune_grid.csv", index=False)
    correlations = pd.DataFrame(
        {
            "comparison": ["original_vs_gainprune", "corrprune_vs_gainprune"],
            "correlation": [
                np.corrcoef(normalized["tabm_mse"], normalized["tabm_gainprune"])[0, 1],
                np.corrcoef(
                    normalized["tabm_corrprune"], normalized["tabm_gainprune"]
                )[0, 1],
            ],
        }
    )
    correlations.to_csv(run_dir / "correlations.csv", index=False)
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "source_files": SOURCE_FILES,
                "weight_grid": GAIN_PRUNED_WEIGHT_GRID,
                "selection": (
                    "audit only; require overall above submitted blend001 and "
                    "simultaneous primary robustness gains versus blend005"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(correlations.to_string(index=False))
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
