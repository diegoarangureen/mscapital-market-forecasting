"""Search four-source weights with a no-regression guard for every primary month."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_tabm_six_tree_pool import (
    integer_compositions,
    precompute_score_terms,
    score_from_terms,
)
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


SOURCE_FILES = {
    "tabm_original_centered": "exp-tabm-001_valid.feather",
    "tabm_corrprune_raw": "exp-tabm-006-corrprune_valid.feather",
    "tabm_cosine_raw": "exp-tabm-003-cosine_valid.feather",
    "xgboost_centered": "exp-tree-053r_valid.feather",
}
BASELINE_WEIGHTS = np.asarray([0.80, 0.00, 0.00, 0.20], dtype=np.float64)
WEIGHT_STEP = 0.05


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_original_centered"]
    validation_rows = reference[["sample_id", "month", "target"]]
    columns = []
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        values = frame["prediction"].to_numpy(dtype=np.float64)
        if name.endswith("_centered"):
            values = values - values.mean()
        columns.append(values / np.linalg.norm(values))
    predictions = np.column_stack(columns)
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    month_values = np.sort(np.unique(months))
    primary_months = [int(month) for month in month_values if month >= 62 and month != 66]
    masks = {
        "overall": np.ones(len(months), dtype=bool),
        "no66": (months >= 62) & (months != 66),
        "recent": months >= 67,
        "early": months <= 64,
    }
    for month in month_values:
        masks[f"month_{int(month)}"] = months == month
    terms = {
        name: precompute_score_terms(target, predictions, mask)
        for name, mask in masks.items()
    }
    baseline_scores = {
        name: score_from_terms(BASELINE_WEIGHTS, term)
        for name, term in terms.items()
    }

    rows = []
    total_units = int(round(1.0 / WEIGHT_STEP))
    for composition in integer_compositions(total_units, len(SOURCE_FILES)):
        weights = np.asarray(composition, dtype=np.float64) * WEIGHT_STEP
        row = {
            f"weight_{name}": float(weight)
            for name, weight in zip(SOURCE_FILES, weights, strict=True)
        }
        for metric in ["overall", "no66", "recent", "early"]:
            row[metric] = score_from_terms(weights, terms[metric])
        month_changes = []
        for month in primary_months:
            score = score_from_terms(weights, terms[f"month_{month}"])
            change = score - baseline_scores[f"month_{month}"]
            row[f"month_{month}_change"] = change
            month_changes.append(change)
        row["primary_month_min_change"] = float(np.min(month_changes))
        row["primary_month_mean_change"] = float(np.mean(month_changes))
        rows.append(row)

    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid["overall"] >= baseline_scores["overall"])
        & (grid["no66"] >= baseline_scores["no66"])
        & (grid["recent"] >= baseline_scores["recent"])
        & (grid["early"] >= baseline_scores["early"] - 0.0002)
        & (grid["primary_month_min_change"] >= 0.0)
    ].copy()
    if not eligible.empty:
        eligible["minimum_key_gain"] = np.minimum(
            eligible["no66"] - baseline_scores["no66"],
            eligible["recent"] - baseline_scores["recent"],
        )
        eligible = eligible.sort_values(
            ["minimum_key_gain", "primary_month_min_change", "overall"],
            ascending=False,
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-009-MONTHWISE-GUARD"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(run_dir / "eligible_candidates.csv", index=False)
    result = {
        "candidate_count": int(len(grid)),
        "eligible_count": int(len(eligible)),
        "weight_step": WEIGHT_STEP,
        "primary_months": primary_months,
        "baseline_weights": BASELINE_WEIGHTS.tolist(),
        "baseline_scores": {
            key: value
            for key, value in baseline_scores.items()
            if key in {"overall", "no66", "recent", "early"}
        },
        "selected_candidate": eligible.iloc[0].to_dict() if not eligible.empty else None,
        "selection_rule": (
            "overall/no66/recent non-decreasing, early tolerance 0.0002, and "
            "every month 62-70 except 66 non-decreasing versus centered 80/20"
        ),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not eligible.empty:
        print(eligible.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
