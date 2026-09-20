"""Audit trimmed TabM member means inside leading blend candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def centered_unit(values: np.ndarray) -> np.ndarray:
    values = values - values.mean()
    return values / np.linalg.norm(values)


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values)


def evaluate(
    target: np.ndarray,
    prediction: np.ndarray,
    months: np.ndarray,
    baseline: np.ndarray,
) -> dict:
    month_values = np.sort(np.unique(months))
    monthly = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    baseline_monthly = np.asarray(
        [
            cosine(target[months == month], baseline[months == month])
            for month in month_values
        ]
    )
    primary_mask = (month_values >= 62) & (month_values != 66)
    primary = monthly[primary_mask]
    primary_changes = monthly[primary_mask] - baseline_monthly[primary_mask]
    lomo = []
    for omitted_month in month_values[primary_mask]:
        mask = (months >= 62) & (months != 66) & (months != omitted_month)
        lomo.append(
            cosine(target[mask], prediction[mask])
            - cosine(target[mask], baseline[mask])
        )
    return {
        "overall": cosine(target, prediction),
        "no66": cosine(
            target[(months >= 62) & (months != 66)],
            prediction[(months >= 62) & (months != 66)],
        ),
        "recent": cosine(target[months >= 67], prediction[months >= 67]),
        "early": cosine(target[months <= 64], prediction[months <= 64]),
        "primary_std": float(primary.std(ddof=0)),
        "primary_worst": float(primary.min()),
        "primary_q25": float(np.quantile(primary, 0.25)),
        "primary_months_improved": int((primary_changes > 0).sum()),
        "primary_month_min_change": float(primary_changes.min()),
        "primary_lomo_min": float(np.min(lomo)),
        "primary_lomo_mean": float(np.mean(lomo)),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    reference = pd.read_feather(prediction_dir / "exp-tabm-001_valid.feather")
    members = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-001"
        / "member_predictions.feather"
    )
    if not np.array_equal(reference["sample_id"].to_numpy(), members["sample_id"].to_numpy()):
        raise AssertionError("Member predictions are not aligned with validation rows.")
    member_columns = [column for column in members if column.startswith("member_")]
    member_matrix = members[member_columns].to_numpy(dtype=np.float64)
    sorted_members = np.sort(member_matrix, axis=1)
    original_predictions = {
        "mean": member_matrix.mean(axis=1),
        "trim1": sorted_members[:, 1:-1].mean(axis=1),
        "trim2": sorted_members[:, 2:-2].mean(axis=1),
    }
    for name in ["trim1", "trim2"]:
        output = reference[["sample_id", "month", "target"]].copy()
        output["prediction"] = original_predictions[name]
        output.to_feather(prediction_dir / f"exp-tabm-001-{name}_valid.feather")

    corr = pd.read_feather(prediction_dir / "exp-tabm-006-corrprune_valid.feather")
    cosine_tabm = pd.read_feather(prediction_dir / "exp-tabm-003-cosine_valid.feather")
    xgboost = pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather")
    validation_rows = reference[["sample_id", "month", "target"]]
    for frame in [corr, cosine_tabm, xgboost]:
        assert_prediction_alignment(frame, validation_rows)
    corr_unit = unit(corr["prediction"].to_numpy(dtype=np.float64))
    cosine_unit = unit(cosine_tabm["prediction"].to_numpy(dtype=np.float64))
    xgboost_unit = centered_unit(xgboost["prediction"].to_numpy(dtype=np.float64))
    original_units = {
        name: centered_unit(values) for name, values in original_predictions.items()
    }

    baselines = {
        "centered80": 0.80 * original_units["mean"] + 0.20 * xgboost_unit,
        "aggressive": (
            0.45 * original_units["mean"]
            + 0.20 * corr_unit
            + 0.20 * cosine_unit
            + 0.15 * xgboost_unit
        ),
        "guarded": (
            0.50 * original_units["mean"]
            + 0.25 * corr_unit
            + 0.05 * cosine_unit
            + 0.20 * xgboost_unit
        ),
    }
    candidates = {}
    for aggregation in ["mean", "trim1", "trim2"]:
        candidates[f"centered80_{aggregation}"] = (
            0.80 * original_units[aggregation] + 0.20 * xgboost_unit
        )
        candidates[f"aggressive_{aggregation}"] = (
            0.45 * original_units[aggregation]
            + 0.20 * corr_unit
            + 0.20 * cosine_unit
            + 0.15 * xgboost_unit
        )
        candidates[f"guarded_{aggregation}"] = (
            0.50 * original_units[aggregation]
            + 0.25 * corr_unit
            + 0.05 * cosine_unit
            + 0.20 * xgboost_unit
        )
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    rows = []
    for name, prediction in candidates.items():
        family = name.rsplit("_", 1)[0]
        rows.append(
            {
                "candidate": name,
                "baseline_family": family,
                **evaluate(target, prediction, months, baselines[family]),
            }
        )
    results = pd.DataFrame(rows)
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-002-BLENDS"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_dir / "trimmed_blend_results.csv", index=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "member_source": "EXP-TABM-MEMBER-001",
                "aggregations": ["mean", "trim1", "trim2"],
                "transformation_uses_labels": False,
                "results": results.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
