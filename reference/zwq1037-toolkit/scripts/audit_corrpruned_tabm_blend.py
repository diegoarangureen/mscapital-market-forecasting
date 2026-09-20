"""Audit small correlated-pruned TabM weights inside EXP-BLEND-003."""

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
    "tabm_cosine": "exp-tabm-003-cosine_valid.feather",
    "xgboost": "exp-tree-053r_valid.feather",
    "lightgbm": "exp-tree-068_valid.feather",
    "histgb": "hist_gradient_boosting_target_centered_level2_valid.feather",
}
PRUNED_WEIGHT_GRID = [0.0, 0.025, 0.05, 0.10, 0.15]


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
    rows_reference = reference[["sample_id", "month", "target"]]
    for frame in frames.values():
        assert_prediction_alignment(frame, rows_reference)

    normalized = {}
    for name, frame in frames.items():
        values = frame["prediction"].to_numpy(dtype=np.float64)
        normalized[name] = values / np.linalg.norm(values)
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    month_values = np.sort(np.unique(months))
    primary_mask = (months >= 62) & (months != 66)
    baseline_prediction = (
        0.55 * normalized["tabm_mse"]
        + 0.15 * normalized["tabm_cosine"]
        + 0.10 * normalized["xgboost"]
        + 0.10 * normalized["lightgbm"]
        + 0.10 * normalized["histgb"]
    )

    output_rows = []
    for pruned_weight in PRUNED_WEIGHT_GRID:
        prediction = (
            (0.55 - pruned_weight) * normalized["tabm_mse"]
            + pruned_weight * normalized["tabm_corrprune"]
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
        output_rows.append(
            {
                "tabm_mse_weight": 0.55 - pruned_weight,
                "tabm_corrprune_weight": pruned_weight,
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
                "primary_lomo_min_change_vs_blend003": float(np.min(lomo_changes)),
                "primary_lomo_mean_change_vs_blend003": float(np.mean(lomo_changes)),
            }
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-005-AUDIT"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(output_rows)
    grid.to_csv(run_dir / "corrprune_grid.csv", index=False)
    correlation = float(
        np.corrcoef(normalized["tabm_mse"], normalized["tabm_corrprune"])[0, 1]
    )
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "source_files": SOURCE_FILES,
                "pruned_weight_grid": PRUNED_WEIGHT_GRID,
                "tabm_prediction_correlation": correlation,
                "selection": "audit only; retain EXP-BLEND-003 unless all robustness gates improve",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"tabm_prediction_correlation={correlation:.10f}")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
