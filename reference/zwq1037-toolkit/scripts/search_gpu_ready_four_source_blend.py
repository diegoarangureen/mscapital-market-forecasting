"""Search a coarse deployable blend of two alternate TabMs plus public-best sources."""

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
WEIGHT_STEP = 0.05
BASELINE_WEIGHTS = {
    "tabm_original_centered": 0.80,
    "tabm_corrprune_raw": 0.00,
    "tabm_cosine_raw": 0.00,
    "xgboost_centered": 0.20,
}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    model_names = list(SOURCE_FILES)
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_original_centered"]
    validation_rows = reference[["sample_id", "month", "target"]]
    normalized_columns = []
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        values = frame["prediction"].to_numpy(dtype=np.float64)
        if name.endswith("_centered"):
            values = values - values.mean()
        normalized_columns.append(values / np.linalg.norm(values))
    predictions = np.column_stack(normalized_columns)
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
    for omitted_month in primary_months:
        masks[f"primary_without_{omitted_month}"] = (
            (months >= 62) & (months != 66) & (months != omitted_month)
        )
    terms = {
        name: precompute_score_terms(target, predictions, mask)
        for name, mask in masks.items()
    }
    baseline_vector = np.asarray(
        [BASELINE_WEIGHTS[name] for name in model_names], dtype=np.float64
    )
    baseline = {
        name: score_from_terms(baseline_vector, term) for name, term in terms.items()
    }
    baseline_monthly = np.asarray(
        [baseline[f"month_{int(month)}"] for month in month_values]
    )
    baseline_primary = baseline_monthly[
        np.asarray([(month >= 62 and month != 66) for month in month_values])
    ]

    rows = []
    total_units = int(round(1.0 / WEIGHT_STEP))
    for composition in integer_compositions(total_units, len(model_names)):
        weights = np.asarray(composition, dtype=np.float64) * WEIGHT_STEP
        row = {
            f"weight_{name}": float(weight)
            for name, weight in zip(model_names, weights, strict=True)
        }
        for metric in ["overall", "no66", "recent", "early"]:
            row[metric] = score_from_terms(weights, terms[metric])
        monthly = np.asarray(
            [
                score_from_terms(weights, terms[f"month_{int(month)}"])
                for month in month_values
            ]
        )
        primary = monthly[
            np.asarray([(month >= 62 and month != 66) for month in month_values])
        ]
        lomo = [
            score_from_terms(weights, terms[f"primary_without_{month}"])
            - baseline[f"primary_without_{month}"]
            for month in primary_months
        ]
        row.update(
            {
                "primary_std": float(primary.std(ddof=0)),
                "primary_worst": float(primary.min()),
                "primary_q25": float(np.quantile(primary, 0.25)),
                "primary_lomo_min": float(np.min(lomo)),
                "primary_lomo_mean": float(np.mean(lomo)),
            }
        )
        rows.append(row)

    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid["overall"] >= baseline["overall"])
        & (grid["no66"] >= baseline["no66"])
        & (grid["recent"] >= baseline["recent"])
        & (grid["early"] >= baseline["early"] - 0.0002)
        & (grid["primary_worst"] >= float(baseline_primary.min()))
        & (grid["primary_q25"] >= float(np.quantile(baseline_primary, 0.25)) - 0.0002)
        & (grid["primary_lomo_min"] >= 0.0)
    ].copy()
    if not eligible.empty:
        eligible["minimum_key_gain"] = np.minimum(
            eligible["no66"] - baseline["no66"],
            eligible["recent"] - baseline["recent"],
        )
        eligible = eligible.sort_values(
            ["minimum_key_gain", "overall", "primary_worst"], ascending=False
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-008-FOUR-SOURCE-SEARCH"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid.sort_values(["no66", "recent", "overall"], ascending=False).to_csv(
        run_dir / "weight_grid.csv", index=False
    )
    eligible.to_csv(run_dir / "eligible_candidates.csv", index=False)
    result = {
        "source_files": SOURCE_FILES,
        "weight_step": WEIGHT_STEP,
        "candidate_count": int(len(grid)),
        "eligible_count": int(len(eligible)),
        "baseline_weights": BASELINE_WEIGHTS,
        "baseline_metrics": {
            "overall": baseline["overall"],
            "no66": baseline["no66"],
            "recent": baseline["recent"],
            "early": baseline["early"],
            "primary_std": float(baseline_primary.std(ddof=0)),
            "primary_worst": float(baseline_primary.min()),
            "primary_q25": float(np.quantile(baseline_primary, 0.25)),
        },
        "selected_candidate": (
            eligible.iloc[0].to_dict() if not eligible.empty else None
        ),
        "selection_rule": (
            "all major metrics and primary LOMO non-decreasing versus centered "
            "80/20 baseline, with early/q25 tolerance 0.0002; maximize minimum "
            "no66/recent gain"
        ),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not eligible.empty:
        print(eligible.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
