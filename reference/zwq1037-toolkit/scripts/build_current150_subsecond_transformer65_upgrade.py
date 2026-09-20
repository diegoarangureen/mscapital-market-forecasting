"""Build the selection-chosen 65% event upgrade for the proven 0.150 blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_current150_subsecond_transformer_upgrade import load, unit


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
RUN_NAME = "current150_subsecond_transformer65_upgrade"
TRANSFORMER_SLOT_WEIGHT = 0.16
EVENT_WEIGHT = 0.65
OLD_WEIGHT = 1.0 - EVENT_WEIGHT
OLD_DEV_RMS = 0.00033945538892300123
EVENT_DEV_RMS = 0.000395204559828322
COMMON_GROUPS = 38


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction))
    )


def main() -> None:
    paths = {
        "base": SUBMISSIONS
        / "factorized_full_replace_corr095_current138k32_equal38_common.csv",
        "old_transformer": PROJECT
        / "data/interim/kaggle_outputs/multistream_factorized_transformer_fulltrain"
        / "factorized_transformer/factorized_transformer379_full_e4.csv",
        "event_model": SUBMISSIONS
        / "standalone_subsecond_event_transformer379_full_e3.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    base = load(paths["base"])
    ids = base["sample_id"].to_numpy()
    old = load(paths["old_transformer"], ids)["prediction"].to_numpy(
        dtype=np.float64
    )
    event = load(paths["event_model"], ids)["prediction"].to_numpy(
        dtype=np.float64
    )
    event_scaled = event * (OLD_DEV_RMS / EVENT_DEV_RMS)
    hybrid = OLD_WEIGHT * old + EVENT_WEIGHT * event_scaled

    delta = TRANSFORMER_SLOT_WEIGHT * (unit(hybrid) - unit(old))
    groups = (np.arange(len(ids), dtype=np.int64) * COMMON_GROUPS // len(ids)).astype(
        np.int16
    )
    for group in np.unique(groups):
        mask = groups == group
        delta[mask] -= delta[mask].mean()

    base_prediction = base["prediction"].to_numpy(dtype=np.float64)
    prediction = base_prediction + delta
    output_path = SUBMISSIONS / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
        output_path, index=False
    )

    report = {
        "run_name": RUN_NAME,
        "base_public_lb": 0.150,
        "base_path": str(paths["base"]),
        "replacement": {
            "slot": "factorized_transformer",
            "slot_weight": TRANSFORMER_SLOT_WEIGHT,
            "old": str(paths["old_transformer"]),
            "event_model": str(paths["event_model"]),
            "new_internal_weights": {
                "old_transformer": OLD_WEIGHT,
                "subsecond_event_model": EVENT_WEIGHT,
            },
            "effective_full_blend_weights": {
                "old_transformer": TRANSFORMER_SLOT_WEIGHT * OLD_WEIGHT,
                "subsecond_event_model": TRANSFORMER_SLOT_WEIGHT * EVENT_WEIGHT,
            },
        },
        "weight_selection": {
            "selected_on_months": [62, 63, 64, 65],
            "grid": "0.00 to 1.00 by 0.05",
            "selected_event_weight": EVENT_WEIGHT,
            "selection_score": 0.1607134936,
            "forward_months": [67, 68, 69, 70],
            "forward_score": 0.1589192691,
            "forward_baseline": 0.1566976834,
            "forward_delta": 0.0022215857,
        },
        "test_diagnostics": {
            "old_vs_hybrid_correlation": float(np.corrcoef(old, hybrid)[0, 1]),
            "candidate_vs_base_correlation": float(
                np.corrcoef(base_prediction, prediction)[0, 1]
            ),
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
