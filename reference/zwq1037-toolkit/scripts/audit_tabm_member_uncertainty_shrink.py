"""Test label-free prediction shrinkage based on TabM member disagreement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_trimmed_tabm_in_blends import centered_unit, evaluate, unit


EXPERIMENT_ID = "EXP-TABM-010-MEMBER-UNCERTAINTY"
MAXIMIN_WEIGHTS = {
    "original": 0.3525,
    "corrprune": 0.20,
    "cosine": 0.1525,
    "xgboost": 0.0975,
    "lightgbm": 0.0975,
    "histgb": 0.10,
}


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return ranks


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    reference = pd.read_feather(prediction_dir / "exp-tabm-001-trim1_valid.feather")
    ids = reference["sample_id"].to_numpy()
    member_frame = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-001"
        / "member_predictions.feather"
    )
    if not np.array_equal(member_frame["sample_id"].to_numpy(), ids):
        raise AssertionError("Member predictions are not aligned.")
    member_columns = [column for column in member_frame if column.startswith("member_")]
    members = member_frame[member_columns].to_numpy(dtype=np.float64)
    sorted_members = np.sort(members, axis=1)
    trim1 = sorted_members[:, 1:-1].mean(axis=1)
    disagreement = members.std(axis=1, ddof=0)
    disagreement_rank = percentile_rank(disagreement)

    cosine_aggregations = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
        / "cosine_aggregations_valid.feather"
    )
    frames = {
        "corrprune": pd.read_feather(
            prediction_dir / "exp-tabm-006-corrprune_valid.feather"
        ),
        "xgboost": pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather"),
        "lightgbm": pd.read_feather(prediction_dir / "exp-tree-068_valid.feather"),
        "histgb": pd.read_feather(
            prediction_dir / "hist_gradient_boosting_target_centered_level2_valid.feather"
        ),
    }
    for name, frame in frames.items():
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError(f"IDs are not aligned for {name}.")
    if not np.array_equal(cosine_aggregations["sample_id"].to_numpy(), ids):
        raise AssertionError("Cosine aggregation IDs are not aligned.")
    other_sources = {
        "corrprune": unit(frames["corrprune"]["prediction"].to_numpy(dtype=np.float64)),
        "cosine": unit(cosine_aggregations["trim1_prediction"].to_numpy(dtype=np.float64)),
        "xgboost": centered_unit(frames["xgboost"]["prediction"].to_numpy(dtype=np.float64)),
        "lightgbm": centered_unit(frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64)),
        "histgb": unit(frames["histgb"]["prediction"].to_numpy(dtype=np.float64)),
    }
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    baseline_original = centered_unit(trim1)
    baseline_six = (
        MAXIMIN_WEIGHTS["original"] * baseline_original
        + sum(MAXIMIN_WEIGHTS[name] * other_sources[name] for name in other_sources)
    )

    candidates: dict[str, np.ndarray] = {"none": trim1}
    for strength in [0.025, 0.05, 0.10, 0.15, 0.20]:
        candidates[f"linear_{strength:.3f}"] = trim1 * (
            1.0 - strength * disagreement_rank
        )
    for strength in [0.05, 0.10, 0.20, 0.30]:
        candidates[f"exp_{strength:.3f}"] = trim1 * np.exp(
            -strength * disagreement_rank
        )
    for quantile, factor in [(0.90, 0.90), (0.90, 0.80), (0.95, 0.80)]:
        confidence = np.ones(len(trim1), dtype=np.float64)
        confidence[disagreement_rank >= quantile] = factor
        candidates[f"top{int((1-quantile)*100):02d}_factor{factor:.2f}"] = trim1 * confidence

    rows = []
    for name, adjusted in candidates.items():
        original = centered_unit(adjusted)
        six = (
            MAXIMIN_WEIGHTS["original"] * original
            + sum(MAXIMIN_WEIGHTS[source] * other_sources[source] for source in other_sources)
        )
        original_metrics = evaluate(target, original, months, baseline_original)
        six_metrics = evaluate(target, six, months, baseline_six)
        rows.append(
            {
                "method": name,
                **{f"original_{key}": value for key, value in original_metrics.items()},
                **{f"six_{key}": value for key, value in six_metrics.items()},
            }
        )
    results = pd.DataFrame(rows)
    baseline = results.loc[results["method"] == "none"].iloc[0]
    robust = results.loc[
        (results["method"] != "none")
        & (results["six_overall"] >= baseline["six_overall"])
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
    results.to_csv(run_dir / "uncertainty_results.csv", index=False)
    robust.to_csv(run_dir / "robust_candidates.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "transformation_uses_labels": False,
        "member_count": int(members.shape[1]),
        "disagreement_summary": {
            "mean": float(disagreement.mean()),
            "std": float(disagreement.std(ddof=0)),
            "correlation_with_abs_trim1": float(np.corrcoef(disagreement, np.abs(trim1))[0, 1]),
        },
        "candidate_count": int(len(results) - 1),
        "robust_candidate_count": int(len(robust)),
        "best_robust": robust.head(1).to_dict(orient="records"),
        "best_six_no66": results.sort_values("six_no66", ascending=False)
        .head(1)
        .to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
