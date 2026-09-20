"""Search a small robust grid around B005 using trimmed TabM sources."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_trimmed_tabm_in_blends import centered_unit, evaluate, unit
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


EXPERIMENT_ID = "EXP-BLEND-011-TRIMMED-SIX-SOURCE"
WEIGHT_GRID = {
    "corrprune": [0.15, 0.20, 0.25],
    "cosine": [0.10, 0.15, 0.20],
    "xgboost": [0.05, 0.10, 0.15],
    "lightgbm": [0.00, 0.05, 0.10, 0.15],
    "histgb": [0.00, 0.05, 0.10],
}
ORIGINAL_WEIGHT_BOUNDS = (0.25, 0.55)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    reference = pd.read_feather(prediction_dir / "exp-tabm-001_valid.feather")
    validation_rows = reference[["sample_id", "month", "target"]]
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()

    original_trim = pd.read_feather(
        prediction_dir / "exp-tabm-001-trim1_valid.feather"
    )
    cosine_aggregations = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
        / "cosine_aggregations_valid.feather"
    )
    source_frames = {
        "corrprune": pd.read_feather(
            prediction_dir / "exp-tabm-006-corrprune_valid.feather"
        ),
        "xgboost": pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather"),
        "lightgbm": pd.read_feather(prediction_dir / "exp-tree-068_valid.feather"),
        "histgb": pd.read_feather(
            prediction_dir / "hist_gradient_boosting_target_centered_level2_valid.feather"
        ),
        "public_best": pd.read_feather(prediction_dir / "exp-blend-001_valid.feather"),
    }
    assert_prediction_alignment(original_trim, validation_rows)
    if not np.array_equal(
        cosine_aggregations["sample_id"].to_numpy(),
        validation_rows["sample_id"].to_numpy(),
    ):
        raise AssertionError("Cosine TabM aggregation rows are not aligned.")
    for frame in source_frames.values():
        assert_prediction_alignment(frame, validation_rows)

    sources = {
        "original": centered_unit(
            original_trim["prediction"].to_numpy(dtype=np.float64)
        ),
        "corrprune": unit(
            source_frames["corrprune"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "cosine": unit(
            cosine_aggregations["trim1_prediction"].to_numpy(dtype=np.float64)
        ),
        "xgboost": centered_unit(
            source_frames["xgboost"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "lightgbm": centered_unit(
            source_frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "histgb": unit(
            source_frames["histgb"]["prediction"].to_numpy(dtype=np.float64)
        ),
    }
    public_best = source_frames["public_best"]["prediction"].to_numpy(
        dtype=np.float64
    )

    rows: list[dict[str, object]] = []
    grid_names = list(WEIGHT_GRID)
    for grid_values in itertools.product(*(WEIGHT_GRID[name] for name in grid_names)):
        weights = dict(zip(grid_names, grid_values))
        original_weight = 1.0 - sum(weights.values())
        if not ORIGINAL_WEIGHT_BOUNDS[0] <= original_weight <= ORIGINAL_WEIGHT_BOUNDS[1]:
            continue
        weights["original"] = original_weight
        prediction = sum(weights[name] * sources[name] for name in sources)
        metrics = evaluate(target, prediction, months, public_best)
        rows.append(
            {
                **{f"weight_{name}": weights[name] for name in sources},
                **metrics,
                "uses_histgb": weights["histgb"] > 0.0,
            }
        )

    results = pd.DataFrame(rows)
    strict = results.loc[
        (results["overall"] >= 0.17571880874787868)
        & (results["no66"] >= 0.1538075419256136)
        & (results["recent"] >= 0.1521286911445723)
        & (results["early"] >= 0.1550)
        & (results["primary_std"] <= 0.0131683400389342)
        & (results["primary_worst"] >= 0.1304067898250277)
        & (results["primary_q25"] >= 0.14722130947635)
        & (results["primary_lomo_min"] > 0.0)
    ].copy()
    results = results.sort_values(
        ["no66", "recent", "primary_worst", "overall"], ascending=False
    )
    strict = strict.sort_values(
        ["no66", "recent", "primary_worst", "overall"], ascending=False
    )

    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_dir / "all_candidates.csv", index=False)
    strict.to_csv(run_dir / "strict_candidates.csv", index=False)
    best_overall = results.sort_values("overall", ascending=False).head(1)
    best_no66 = results.head(1)
    best_no_hist = results.loc[~results["uses_histgb"]].head(1)
    selected = {
        "best_overall": best_overall.to_dict(orient="records"),
        "best_no66": best_no66.to_dict(orient="records"),
        "best_no_histgb": best_no_hist.to_dict(orient="records"),
        "best_strict": strict.head(10).to_dict(orient="records"),
    }
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "aggregation_uses_labels": False,
        "grid": WEIGHT_GRID,
        "original_weight_bounds": ORIGINAL_WEIGHT_BOUNDS,
        "candidate_count": int(len(results)),
        "strict_candidate_count": int(len(strict)),
        "strict_gate_reference": "double-trim aggressive four-source candidate",
        "selection": selected,
    }
    (run_dir / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
