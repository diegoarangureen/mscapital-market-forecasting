"""Search a coarse, selection-only simplex for the owned Transformer slot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = ROOT / "data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE"
RAW_PATH = ROOT / "data/interim/tree_experiments/EXP-BLEND-004-EVENT-RESIDUAL-SOFTGATE/transformer020_raw_candidate.feather"
EVENT_PATH = ROOT / "data/interim/kaggle_results/market_conditioned_event_v24/market_conditioned_event_residual/validation_predictions.feather"
OUTPUT = ROOT / "outputs/submission_metadata/transformer020_event_joint_softgate_search.json"
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


def load_candidate(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_feather(path)[["sample_id", "base_prediction", "prediction"]]
    return frame.rename(
        columns={
            "base_prediction": f"{prefix}_base_prediction",
            "prediction": f"{prefix}_prediction",
        }
    )


def main() -> None:
    frame = pd.read_feather(BLEND_DIR / "validation_predictions.feather")
    frame = frame.merge(load_candidate(RAW_PATH, "raw"), on="sample_id", how="inner", validate="one_to_one")
    frame = frame.merge(load_candidate(EVENT_PATH, "event"), on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != 140_806:
        raise AssertionError(f"Validation alignment failed: {len(frame)}")

    target = frame.target.to_numpy(np.float64)
    months = frame.month.to_numpy()
    selection = np.isin(months, [62, 63, 64, 65])
    old_transformer = frame.transformer.to_numpy(np.float64)
    reproductions = {
        "raw": cosine(old_transformer, frame.raw_base_prediction.to_numpy(np.float64)),
        "event": cosine(old_transformer, frame.event_base_prediction.to_numpy(np.float64)),
    }
    if min(reproductions.values()) < 0.999999:
        raise AssertionError(f"Transformer baseline mismatch: {reproductions}")

    parameters = np.load(BLEND_DIR / "fusion_parameters.npz")
    scale = parameters["scale"].astype(np.float64)
    anchors = parameters["anchors"].astype(np.float64)
    state_weights = parameters["state_weights"].astype(np.float64)
    rv_median = float(parameters["rv_median"])

    raw_models = frame[NAMES].to_numpy(np.float64)
    standardized = raw_models / scale
    old_standardized = standardized[:, NAMES.index("transformer")]
    candidate_standardized = {}
    candidate_scales = {}
    for candidate_name in ("raw", "event"):
        values = frame[f"{candidate_name}_prediction"].to_numpy(np.float64)
        rms = float(np.sqrt(np.mean(values[selection] ** 2)))
        candidate_scales[candidate_name] = rms
        candidate_standardized[candidate_name] = values / max(rms, 1e-12)

    if "realized_volatility_60" not in frame.columns:
        volatility_frame = pd.read_feather(
            ROOT / "data/processed/train_market_microstructure_features.feather",
            columns=["sample_id", "realized_volatility_60"],
        )
        frame = frame.merge(volatility_frame, on="sample_id", how="left", validate="one_to_one")
    volatility = np.nan_to_num(
        frame.realized_volatility_60.to_numpy(np.float64),
        nan=rv_median,
        posinf=rv_median,
        neginf=rv_median,
    )
    sample_weights = np.column_stack(
        [np.interp(volatility, anchors, state_weights[:, column]) for column in range(len(NAMES))]
    )
    if not np.allclose(sample_weights.sum(axis=1), 1.0):
        raise AssertionError("Soft-gate weights do not sum to one")

    baseline = (standardized * sample_weights).sum(axis=1)
    baseline_metrics = metrics(target, baseline, months)
    transformer_weight = sample_weights[:, NAMES.index("transformer")]

    rows = []
    step = 0.05
    units = int(round(1.0 / step))
    for raw_units in range(units + 1):
        for event_units in range(units - raw_units + 1):
            raw_weight = raw_units * step
            event_weight = event_units * step
            old_weight = 1.0 - raw_weight - event_weight
            slot = (
                old_weight * old_standardized
                + raw_weight * candidate_standardized["raw"]
                + event_weight * candidate_standardized["event"]
            )
            prediction = baseline + transformer_weight * (slot - old_standardized)
            current = metrics(target, prediction, months)
            monthly_delta = {
                month: current["monthly_forward"][month] - baseline_metrics["monthly_forward"][month]
                for month in baseline_metrics["monthly_forward"]
            }
            rows.append(
                {
                    "old_weight": old_weight,
                    "raw_weight": raw_weight,
                    "event_weight": event_weight,
                    **current,
                    "selection_delta": current["selection"] - baseline_metrics["selection"],
                    "forward_delta": current["forward"] - baseline_metrics["forward"],
                    "all_delta": current["all"] - baseline_metrics["all"],
                    "monthly_delta": monthly_delta,
                }
            )

    selected = max(rows, key=lambda item: item["selection"])
    monthly_delta = selected["monthly_delta"]
    passed = bool(
        selected["selection_delta"] >= 0.0005
        and selected["forward_delta"] >= 0.0005
        and selected["all_delta"] >= 0.0007
        and sum(value >= 0 for value in monthly_delta.values()) >= 3
        and min(monthly_delta.values()) >= -0.002
    )
    top_selection = sorted(rows, key=lambda item: item["selection"], reverse=True)[:10]
    report = {
        "experiment": "EXP-BLEND-005-TRANSFORMER020-EVENT-JOINT-SIMPLEX",
        "selection_rule": "Choose the highest cosine on months 62-65 only from a 0.05 simplex grid.",
        "baseline": baseline_metrics,
        "candidate_reproduction_cosine": reproductions,
        "candidate_selection_rms": candidate_scales,
        "selected_by_62_65": selected,
        "top10_by_selection": top_selection,
        "passed": passed,
        "gate": {
            "selection_delta_min": 0.0005,
            "forward_delta_min": 0.0005,
            "all_delta_min": 0.0007,
            "forward_months_nonnegative_min": 3,
            "worst_forward_month_delta_min": -0.002,
        },
        "grid_size": len(rows),
        "note": "Public component is unavailable offline; only the owned Transformer slot is changed.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "selected": selected, "grid_size": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

