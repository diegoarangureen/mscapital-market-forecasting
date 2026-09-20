"""Audit the frozen 40/60 raw-event slot with leave-one-selection-month-out checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_transformer020_event_joint_softgate import (
    BLEND_DIR,
    EVENT_PATH,
    NAMES,
    RAW_PATH,
    ROOT,
    cosine,
    load_candidate,
)


OUTPUT = ROOT / "outputs/submission_metadata/transformer020_event_joint_lomo_audit.json"


def main() -> None:
    frame = pd.read_feather(BLEND_DIR / "validation_predictions.feather")
    frame = frame.merge(load_candidate(RAW_PATH, "raw"), on="sample_id", how="inner", validate="one_to_one")
    frame = frame.merge(load_candidate(EVENT_PATH, "event"), on="sample_id", how="inner", validate="one_to_one")
    target = frame.target.to_numpy(np.float64)
    months = frame.month.to_numpy()
    selection = np.isin(months, [62, 63, 64, 65])

    parameters = np.load(BLEND_DIR / "fusion_parameters.npz")
    scale = parameters["scale"].astype(np.float64)
    anchors = parameters["anchors"].astype(np.float64)
    state_weights = parameters["state_weights"].astype(np.float64)
    rv_median = float(parameters["rv_median"])

    standardized = frame[NAMES].to_numpy(np.float64) / scale
    old = standardized[:, 2]
    candidates = {}
    for name in ("raw", "event"):
        values = frame[f"{name}_prediction"].to_numpy(np.float64)
        candidates[name] = values / np.sqrt(np.mean(values[selection] ** 2))

    if "realized_volatility_60" not in frame.columns:
        rv = pd.read_feather(
            ROOT / "data/processed/train_market_microstructure_features.feather",
            columns=["sample_id", "realized_volatility_60"],
        )
        frame = frame.merge(rv, on="sample_id", how="left", validate="one_to_one")
    volatility = np.nan_to_num(
        frame.realized_volatility_60.to_numpy(np.float64),
        nan=rv_median,
        posinf=rv_median,
        neginf=rv_median,
    )
    sample_weights = np.column_stack(
        [np.interp(volatility, anchors, state_weights[:, column]) for column in range(len(NAMES))]
    )
    baseline = (standardized * sample_weights).sum(axis=1)
    transformer_weight = sample_weights[:, 2]

    def prediction(raw_weight: float) -> np.ndarray:
        slot = raw_weight * candidates["raw"] + (1.0 - raw_weight) * candidates["event"]
        return baseline + transformer_weight * (slot - old)

    frozen = prediction(0.4)
    rows = []
    for held_out in (62, 63, 64, 65):
        fit_mask = selection & (months != held_out)
        held_mask = months == held_out
        grid = []
        for raw_weight in np.linspace(0.0, 1.0, 11):
            current = prediction(float(raw_weight))
            grid.append(
                {
                    "raw_weight": float(raw_weight),
                    "fit_score": cosine(target[fit_mask], current[fit_mask]),
                }
            )
        chosen = max(grid, key=lambda row: row["fit_score"])
        chosen_prediction = prediction(chosen["raw_weight"])
        rows.append(
            {
                "held_out_month": held_out,
                "three_month_selected_raw_weight": chosen["raw_weight"],
                "held_out_delta_selected": cosine(target[held_mask], chosen_prediction[held_mask])
                - cosine(target[held_mask], baseline[held_mask]),
                "held_out_delta_frozen_40_60": cosine(target[held_mask], frozen[held_mask])
                - cosine(target[held_mask], baseline[held_mask]),
            }
        )

    report = {
        "experiment": "EXP-BLEND-005-TRANSFORMER020-EVENT-JOINT-LOMO-AUDIT",
        "frozen_weights": {"raw": 0.4, "event": 0.6},
        "rows": rows,
        "frozen_positive_held_out_months": sum(row["held_out_delta_frozen_40_60"] >= 0 for row in rows),
        "selected_positive_held_out_months": sum(row["held_out_delta_selected"] >= 0 for row in rows),
        "selected_weight_range": [
            min(row["three_month_selected_raw_weight"] for row in rows),
            max(row["three_month_selected_raw_weight"] for row in rows),
        ],
        "interpretation": "Weights remain frozen at 40/60; this audit measures whether their gain is concentrated in one selection month.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

