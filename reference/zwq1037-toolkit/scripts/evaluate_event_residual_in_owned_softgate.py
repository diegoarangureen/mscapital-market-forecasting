"""Evaluate a Transformer event-residual replacement inside the owned soft gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = ROOT / "data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE"
NAMES = ["tabm", "realmlp", "transformer", "gru"]


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(target @ prediction / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30))


def metrics(target: np.ndarray, prediction: np.ndarray, months: np.ndarray) -> dict:
    selection = np.isin(months, [62, 63, 64, 65])
    forward = np.isin(months, [67, 68, 69, 70])
    return {
        "selection": cosine(target[selection], prediction[selection]),
        "forward": cosine(target[forward], prediction[forward]),
        "all": cosine(target, prediction),
        "monthly_forward": {
            str(month): cosine(target[months == month], prediction[months == month])
            for month in range(67, 71)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_feather(BLEND_DIR / "validation_predictions.feather")
    candidate = pd.read_feather(args.candidate)[["sample_id", "base_prediction", "prediction"]]
    frame = frame.merge(candidate, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != 140_806:
        raise AssertionError(f"Validation alignment failed: {len(frame)}")

    target = frame.target.to_numpy(np.float64)
    months = frame.month.to_numpy()
    selection = np.isin(months, [62, 63, 64, 65])
    old_transformer = frame.transformer.to_numpy(np.float64)
    reproduced = frame.base_prediction.to_numpy(np.float64)
    reproduction_cosine = cosine(old_transformer, reproduced)
    reproduction_rmse = float(np.sqrt(np.mean((old_transformer - reproduced) ** 2)))
    if reproduction_cosine < 0.999999:
        raise AssertionError(f"Transformer baseline mismatch: cosine={reproduction_cosine}")

    parameters = np.load(BLEND_DIR / "fusion_parameters.npz")
    scale = parameters["scale"].astype(np.float64)
    anchors = parameters["anchors"].astype(np.float64)
    state_weights = parameters["state_weights"].astype(np.float64)
    rv_median = float(parameters["rv_median"])
    raw = frame[NAMES].to_numpy(np.float64)
    standardized = raw / scale
    candidate_values = frame.prediction.to_numpy(np.float64)
    candidate_scale = float(np.sqrt(np.mean(candidate_values[selection] ** 2)))
    candidate_standardized = candidate_values / max(candidate_scale, 1e-12)

    if "realized_volatility_60" not in frame.columns:
        rv = pd.read_feather(
            ROOT / "data/processed/train_market_microstructure_features.feather",
            columns=["sample_id", "realized_volatility_60"],
        )
        frame = frame.merge(rv, on="sample_id", how="left", validate="one_to_one")
    volatility = np.nan_to_num(
        frame.realized_volatility_60.to_numpy(np.float64),
        nan=rv_median, posinf=rv_median, neginf=rv_median,
    )
    sample_weights = np.column_stack([
        np.interp(volatility, anchors, state_weights[:, column])
        for column in range(len(NAMES))
    ])
    if not np.allclose(sample_weights.sum(axis=1), 1.0):
        raise AssertionError("Soft-gate weights do not sum to one")
    baseline = (standardized * sample_weights).sum(axis=1)
    baseline_metrics = metrics(target, baseline, months)
    transformer_weight = sample_weights[:, NAMES.index("transformer")]

    rows = []
    for replacement_fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        prediction = baseline + transformer_weight * replacement_fraction * (
            candidate_standardized - standardized[:, NAMES.index("transformer")]
        )
        current = metrics(target, prediction, months)
        current["replacement_fraction"] = replacement_fraction
        current["selection_delta"] = current["selection"] - baseline_metrics["selection"]
        current["forward_delta"] = current["forward"] - baseline_metrics["forward"]
        current["monthly_delta"] = {
            month: current["monthly_forward"][month] - baseline_metrics["monthly_forward"][month]
            for month in baseline_metrics["monthly_forward"]
        }
        rows.append(current)

    selected = max(rows, key=lambda item: item["selection"])
    monthly_delta = selected["monthly_delta"]
    passed = bool(
        selected["replacement_fraction"] > 0
        and selected["selection_delta"] >= 0
        and selected["forward_delta"] >= 0.0003
        and sum(value >= 0 for value in monthly_delta.values()) >= 3
        and min(monthly_delta.values()) >= -0.0005
    )
    report = {
        "experiment": "EXP-BLEND-004-EVENT-RESIDUAL-SOFTGATE-REPLACEMENT",
        "candidate": str(args.candidate),
        "transformer_reproduction_cosine": reproduction_cosine,
        "transformer_reproduction_rmse": reproduction_rmse,
        "candidate_selection_rms": candidate_scale,
        "baseline": baseline_metrics,
        "rows": rows,
        "selected_by_62_65": selected,
        "passed": passed,
        "gate": {
            "selection_delta_min": 0.0,
            "forward_delta_min": 0.0003,
            "forward_months_nonnegative_min": 3,
            "worst_forward_month_delta_min": -0.0005,
        },
        "interpretation": "Replace only the owned Transformer slot inside the saved volatility soft gate; public block is unavailable offline.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "selected": selected, "reproduction_cosine": reproduction_cosine}), flush=True)


if __name__ == "__main__":
    main()
