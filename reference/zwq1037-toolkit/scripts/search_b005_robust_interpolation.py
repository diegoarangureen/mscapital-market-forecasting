"""Search a one-dimensional interpolation between B005 and robust weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_trimmed_tabm_in_blends import centered_unit, cosine, evaluate, unit


EXPERIMENT_ID = "EXP-BLEND-014-B005-ROBUST-INTERPOLATION"
SOURCE_NAMES = ["original", "corrprune", "cosine", "xgboost", "lightgbm", "histgb"]
B005 = np.asarray([0.35, 0.20, 0.15, 0.10, 0.10, 0.10], dtype=np.float64)
ROBUST = np.asarray([0.40, 0.20, 0.20, 0.05, 0.05, 0.10], dtype=np.float64)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    reference = pd.read_feather(prediction_dir / "exp-tabm-001-trim1_valid.feather")
    ids = reference["sample_id"].to_numpy()
    cosine_aggregations = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
        / "cosine_aggregations_valid.feather"
    )
    frames = {
        "original": reference,
        "corrprune": pd.read_feather(
            prediction_dir / "exp-tabm-006-corrprune_valid.feather"
        ),
        "cosine": cosine_aggregations,
        "xgboost": pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather"),
        "lightgbm": pd.read_feather(prediction_dir / "exp-tree-068_valid.feather"),
        "histgb": pd.read_feather(
            prediction_dir / "hist_gradient_boosting_target_centered_level2_valid.feather"
        ),
        "public_best": pd.read_feather(prediction_dir / "exp-blend-001_valid.feather"),
    }
    for name, frame in frames.items():
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError(f"Validation IDs are not aligned for {name}.")
    source_matrix = np.column_stack(
        [
            centered_unit(frames["original"]["prediction"].to_numpy(dtype=np.float64)),
            unit(frames["corrprune"]["prediction"].to_numpy(dtype=np.float64)),
            unit(frames["cosine"]["trim1_prediction"].to_numpy(dtype=np.float64)),
            centered_unit(frames["xgboost"]["prediction"].to_numpy(dtype=np.float64)),
            centered_unit(frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64)),
            unit(frames["histgb"]["prediction"].to_numpy(dtype=np.float64)),
        ]
    )
    public_best = frames["public_best"]["prediction"].to_numpy(dtype=np.float64)
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    primary_months = np.asarray([62, 63, 64, 65, 67, 68, 69, 70])
    baseline_monthly = {
        int(month): cosine(
            target[months == month], public_best[months == month]
        )
        for month in primary_months
    }

    rows = []
    for alpha in np.linspace(0.0, 1.0, 101):
        weights = (1.0 - alpha) * B005 + alpha * ROBUST
        prediction = source_matrix @ weights
        metrics = evaluate(target, prediction, months, public_best)
        primary_changes = {
            int(month): (
                cosine(target[months == month], prediction[months == month])
                - baseline_monthly[int(month)]
            )
            for month in primary_months
        }
        rows.append(
            {
                "alpha_robust": float(alpha),
                **{f"weight_{name}": float(value) for name, value in zip(SOURCE_NAMES, weights)},
                **metrics,
                "maximin_primary_month_change": float(min(primary_changes.values())),
                "maximin_month": int(min(primary_changes, key=primary_changes.get)),
                "month65_change": primary_changes[65],
                "month69_change": primary_changes[69],
            }
        )
    results = pd.DataFrame(rows)
    maximin = results.sort_values(
        ["maximin_primary_month_change", "no66", "recent"], ascending=False
    ).head(1)
    b005 = results.loc[np.isclose(results["alpha_robust"], 0.0)].iloc[0]
    balanced = results.loc[
        (results["overall"] >= b005["overall"])
        & (results["no66"] >= b005["no66"])
        & (results["recent"] >= b005["recent"])
        & (results["early"] >= b005["early"])
        & (results["primary_std"] <= b005["primary_std"] + 0.00005)
        & (results["primary_worst"] >= b005["primary_worst"])
        & (results["primary_q25"] >= b005["primary_q25"] - 0.00010)
        & (
            results["maximin_primary_month_change"]
            >= b005["maximin_primary_month_change"]
        )
    ].sort_values(["no66", "recent"], ascending=False)

    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_dir / "interpolation_grid.csv", index=False)
    balanced.to_csv(run_dir / "balanced_candidates.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "search_dimension": "linear interpolation only",
        "alpha_grid": "0.00 to 1.00 by 0.01",
        "b005_weights": dict(zip(SOURCE_NAMES, B005.tolist())),
        "robust_weights": dict(zip(SOURCE_NAMES, ROBUST.tolist())),
        "maximin_candidate": maximin.to_dict(orient="records"),
        "balanced_candidate_count": int(len(balanced)),
        "best_balanced": balanced.head(1).to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
