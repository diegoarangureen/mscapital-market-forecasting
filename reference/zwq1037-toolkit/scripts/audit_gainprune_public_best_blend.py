"""Audit adding fulltrain-submitted gain-pruned TabM to the public-best blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


GAIN_WEIGHT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
SOURCE_FILES = {
    "tabm_original": "exp-tabm-001_valid.feather",
    "tabm_gainprune": "exp-tabm-007-gainprune_valid.feather",
    "xgboost": "exp-tree-053r_valid.feather",
}


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
    reference = frames["tabm_original"]
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
    baseline = 0.75 * normalized["tabm_original"] + 0.25 * normalized["xgboost"]
    rows = []
    for gain_weight in GAIN_WEIGHT_GRID:
        prediction = (
            (0.75 - gain_weight) * normalized["tabm_original"]
            + gain_weight * normalized["tabm_gainprune"]
            + 0.25 * normalized["xgboost"]
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
        primary_lomo = []
        for omitted_month in month_values[(month_values >= 62) & (month_values != 66)]:
            mask = primary_mask & (months != omitted_month)
            primary_lomo.append(
                cosine(target[mask], prediction[mask])
                - cosine(target[mask], baseline[mask])
            )
        rows.append(
            {
                "tabm_original_weight": 0.75 - gain_weight,
                "tabm_gainprune_weight": gain_weight,
                "xgboost_weight": 0.25,
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
                "primary_lomo_min_change": float(np.min(primary_lomo)),
                "primary_lomo_mean_change": float(np.mean(primary_lomo)),
            }
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-007-AUDIT"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(run_dir / "gainprune_public_best_grid.csv", index=False)
    correlation = float(
        np.corrcoef(normalized["tabm_original"], normalized["tabm_gainprune"])[0, 1]
    )
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "source_files": SOURCE_FILES,
                "weight_grid": GAIN_WEIGHT_GRID,
                "tabm_prediction_correlation": correlation,
                "public_scores": {
                    "baseline_blend": 0.131,
                    "gainpruned_tabm": 0.130,
                },
                "submission_policy": "audit and prepare only; do not upload without user request",
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
