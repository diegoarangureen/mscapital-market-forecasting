"""Audit seed137 as a diversity source for TabM and the leading six-source blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_trimmed_tabm_in_blends import centered_unit, evaluate, unit


EXPERIMENT_ID = "EXP-TABM-009-MULTISEED-BLEND"
ALPHAS = np.linspace(0.0, 0.50, 21)
MAXIMIN_WEIGHTS = {
    "original": 0.3525,
    "corrprune": 0.20,
    "cosine": 0.1525,
    "xgboost": 0.0975,
    "lightgbm": 0.0975,
    "histgb": 0.10,
}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    seed42 = pd.read_feather(prediction_dir / "exp-tabm-001-trim1_valid.feather")
    seed137 = pd.read_feather(prediction_dir / "exp-tabm-008-seed137_valid.feather")
    validation_rows = seed42[["sample_id", "month", "target"]]
    ids = validation_rows["sample_id"].to_numpy()
    frames = {
        "seed137": seed137,
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
    cosine_aggregations = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
        / "cosine_aggregations_valid.feather"
    )
    for name, frame in frames.items():
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError(f"IDs are not aligned for {name}.")
    if not np.array_equal(cosine_aggregations["sample_id"].to_numpy(), ids):
        raise AssertionError("Cosine aggregation IDs are not aligned.")

    seed42_unit = centered_unit(seed42["prediction"].to_numpy(dtype=np.float64))
    seed137_unit = centered_unit(seed137["prediction"].to_numpy(dtype=np.float64))
    sources = {
        "corrprune": unit(
            frames["corrprune"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "cosine": unit(
            cosine_aggregations["trim1_prediction"].to_numpy(dtype=np.float64)
        ),
        "xgboost": centered_unit(
            frames["xgboost"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "lightgbm": centered_unit(
            frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "histgb": unit(
            frames["histgb"]["prediction"].to_numpy(dtype=np.float64)
        ),
    }
    target = validation_rows["target"].to_numpy(dtype=np.float64)
    months = validation_rows["month"].to_numpy()
    public_best = frames["public_best"]["prediction"].to_numpy(dtype=np.float64)

    baseline_original = seed42_unit
    baseline_six = (
        MAXIMIN_WEIGHTS["original"] * seed42_unit
        + sum(MAXIMIN_WEIGHTS[name] * sources[name] for name in sources)
    )
    rows = []
    for alpha in ALPHAS:
        original_composite = centered_unit(
            (1.0 - alpha) * seed42_unit + alpha * seed137_unit
        )
        six_prediction = (
            MAXIMIN_WEIGHTS["original"] * original_composite
            + sum(MAXIMIN_WEIGHTS[name] * sources[name] for name in sources)
        )
        original_metrics = evaluate(
            target, original_composite, months, baseline_original
        )
        six_metrics = evaluate(target, six_prediction, months, baseline_six)
        rows.append(
            {
                "alpha_seed137": float(alpha),
                **{f"original_{key}": value for key, value in original_metrics.items()},
                **{f"six_{key}": value for key, value in six_metrics.items()},
                "six_no66_vs_public_best": (
                    six_metrics["no66"]
                    - evaluate(target, public_best, months, public_best)["no66"]
                ),
            }
        )
    results = pd.DataFrame(rows)
    baseline = results.iloc[0]
    robust = results.loc[
        (results["six_overall"] >= baseline["six_overall"])
        & (results["six_no66"] >= baseline["six_no66"])
        & (results["six_recent"] >= baseline["six_recent"])
        & (results["six_early"] >= baseline["six_early"] - 0.00010)
        & (results["six_primary_std"] <= baseline["six_primary_std"] + 0.00010)
        & (results["six_primary_worst"] >= baseline["six_primary_worst"])
        & (results["six_primary_q25"] >= baseline["six_primary_q25"] - 0.00010)
        & (results["six_primary_lomo_min"] >= 0.0)
    ].sort_values(["six_no66", "six_recent", "six_overall"], ascending=False)

    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_dir / "alpha_grid.csv", index=False)
    robust.to_csv(run_dir / "robust_candidates.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "seed42_seed137_prediction_correlation": float(
            np.corrcoef(seed42_unit, seed137_unit)[0, 1]
        ),
        "alpha_grid": ALPHAS.tolist(),
        "robust_candidate_count": int(len(robust)),
        "best_robust": robust.head(1).to_dict(orient="records"),
        "best_six_no66": results.sort_values("six_no66", ascending=False)
        .head(1)
        .to_dict(orient="records"),
        "best_original_no66": results.sort_values("original_no66", ascending=False)
        .head(1)
        .to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
