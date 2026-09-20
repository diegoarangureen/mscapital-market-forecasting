"""Upgrade the remaining old-Transformer half inside the scored 0.151 blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factorized_transformer_lb_candidates import build_common_component


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
RUN_NAME = "current151_multiwindow50_subsecond40_remaininghalf"
TRANSFORMER_SLOT_WEIGHT = 0.18
REMAINING_OLD_FRACTION = 0.50
EVENT_WEIGHT = 0.40
OLD_DEV_RMS = 0.00033945538892300123
EVENT_DEV_RMS = 0.000395204559828322


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"] or len(frame) != len(
        expected_ids
    ):
        raise AssertionError(f"Invalid submission shape: {path}")
    if not frame["sample_id"].is_unique or not np.array_equal(
        frame["sample_id"].to_numpy(), expected_ids
    ):
        raise AssertionError(f"Invalid IDs: {path}")
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError(f"Nonfinite predictions: {path}")
    return prediction


def main() -> None:
    _, template, groups, _ = build_common_component()
    ids = template["sample_id"].to_numpy()
    paths = {
        "base": SUBMISSIONS / "current151_structure_transformer50.csv",
        "old_transformer": SUBMISSIONS / "standalone_factorized_transformer379.csv",
        "event_model": SUBMISSIONS
        / "standalone_subsecond_event_transformer379_full_e3.csv",
    }
    base = load(paths["base"], ids)
    old = load(paths["old_transformer"], ids)
    event = load(paths["event_model"], ids)
    hybrid = (1.0 - EVENT_WEIGHT) * old + EVENT_WEIGHT * event * (
        OLD_DEV_RMS / EVENT_DEV_RMS
    )

    replacement_weight = TRANSFORMER_SLOT_WEIGHT * REMAINING_OLD_FRACTION
    delta = replacement_weight * (unit(hybrid) - unit(old))
    for group in np.unique(groups):
        mask = groups == group
        delta[mask] -= delta[mask].mean()
    prediction = base + delta

    output_path = SUBMISSIONS / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
        output_path, index=False
    )
    report = {
        "run_name": RUN_NAME,
        "base_public_lb": 0.151,
        "base_submission_ref": 56275973,
        "base_path": str(paths["base"]),
        "operation": {
            "transformer_slot_weight": TRANSFORMER_SLOT_WEIGHT,
            "already_multiwindow_fraction": 0.50,
            "replaced_remaining_old_fraction": REMAINING_OLD_FRACTION,
            "replacement_weight_in_full_blend": replacement_weight,
            "hybrid_internal_event_weight": EVENT_WEIGHT,
            "effective_event_weight_in_full_blend": replacement_weight
            * EVENT_WEIGHT,
        },
        "validation_evidence": {
            "selection_62_65_delta": 0.0065568491,
            "forward_67_70_delta": 0.0037390426,
            "all_62_70_without_66_delta": 0.0046355864,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "test_diagnostics": {
            "old_vs_hybrid_correlation": float(np.corrcoef(old, hybrid)[0, 1]),
            "candidate_vs_base_correlation": float(np.corrcoef(base, prediction)[0, 1]),
            "delta_std": float(delta.std()),
            "rows": len(ids),
            "finite": bool(np.isfinite(prediction).all()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
        },
        "output_path": str(output_path),
        "submission_status": "prepared_not_submitted",
    }
    (METADATA / f"{RUN_NAME}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
