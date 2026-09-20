"""Upgrade TabM and RealMLP, and half-upgrade the tied Transformer slot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
RUN_NAME = "current150_temporal_balanced_tabm_realmlp57_transformer50"
SLOT_WEIGHTS = {"current": 0.15, "realmlp": 0.05, "transformer": 0.16}
REPLACEMENT_FRACTIONS = {"current": 1.0, "realmlp": 1.0, "transformer": 0.5}


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if len(frame) != 647_896 or not frame["sample_id"].is_unique:
        raise AssertionError(f"Invalid rows or IDs: {path}")
    if ids is not None and not np.array_equal(frame["sample_id"].to_numpy(), ids):
        raise AssertionError(f"IDs do not align: {path}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Nonfinite predictions: {path}")
    return frame


def main() -> None:
    paths = {
        "base": SUBMISSIONS / "factorized_full_replace_corr095_current138k32_equal38_common.csv",
        "old_current": SUBMISSIONS / "current138_refreshed_tabm385_k32.csv",
        "new_current": SUBMISSIONS / "current138_refreshed_tabm385_temporal3x3.csv",
        "old_realmlp": SUBMISSIONS / "standalone_realmlp379_corr095.csv",
        "new_realmlp": SUBMISSIONS / "realmlp379_rq16_temporal3fold_e8_fold_end57.csv",
        "old_transformer": SUBMISSIONS / "standalone_factorized_transformer379.csv",
        "new_transformer": SUBMISSIONS / "standalone_factorized_transformer379_temporal3fold.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing balanced-upgrade inputs:\n" + "\n".join(missing))

    base = load(paths["base"])
    ids = base["sample_id"].to_numpy()
    frames = {name: load(path, ids) for name, path in paths.items() if name != "base"}
    delta = np.zeros(len(ids), dtype=np.float64)
    correlations = {}
    for slot, slot_weight in SLOT_WEIGHTS.items():
        old = frames[f"old_{slot}"]["prediction"].to_numpy(dtype=np.float64)
        new = frames[f"new_{slot}"]["prediction"].to_numpy(dtype=np.float64)
        fraction = REPLACEMENT_FRACTIONS[slot]
        delta += slot_weight * fraction * (unit(new) - unit(old))
        correlations[slot] = float(np.corrcoef(old, new)[0, 1])

    groups = (np.arange(len(ids), dtype=np.int64) * 38 // len(ids)).astype(np.int16)
    for group in np.unique(groups):
        mask = groups == group
        delta[mask] -= delta[mask].mean()
    prediction = base["prediction"].to_numpy(dtype=np.float64) + delta
    output_path = SUBMISSIONS / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
        output_path, index=False
    )
    report = {
        "run_name": RUN_NAME,
        "base_public_lb": 0.150,
        "base_path": str(paths["base"]),
        "slot_weights": SLOT_WEIGHTS,
        "replacement_fractions": REPLACEMENT_FRACTIONS,
        "evidence": {
            "tabm_old_public_lb": 0.134,
            "tabm_temporal_public_lb": 0.140,
            "realmlp57_local_delta_62_70_without_66": 0.00092354109266005,
            "transformer_old_public_lb": 0.145,
            "transformer_temporal_public_lb": 0.145,
        },
        "old_vs_new_test_correlation": correlations,
        "candidate": {
            "rows": len(ids),
            "finite": bool(np.isfinite(prediction).all()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "correlation_with_current_best": float(
                np.corrcoef(base["prediction"], prediction)[0, 1]
            ),
        },
        "output_path": str(output_path),
        "submission_status": "prepared_not_submitted",
        "submission_policy": "preferred no-GRU final candidate after tied Transformer calibration",
    }
    (METADATA / f"{RUN_NAME}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
