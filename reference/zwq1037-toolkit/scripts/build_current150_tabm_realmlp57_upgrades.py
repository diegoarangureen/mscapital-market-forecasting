"""Replace the owned TabM and RealMLP slots in the current 0.150 blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
RUN_NAME = "current150_tabm_temporal_realmlp57_upgrades"
WEIGHTS = {"current": 0.15, "realmlp": 0.05}


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
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
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing temporal upgrade files:\n" + "\n".join(missing))

    base = load(paths["base"])
    ids = base["sample_id"].to_numpy()
    frames = {name: load(path, ids) for name, path in paths.items() if name != "base"}
    delta = np.zeros(len(ids), dtype=np.float64)
    correlations = {}
    for slot in WEIGHTS:
        old = frames[f"old_{slot}"]["prediction"].to_numpy(dtype=np.float64)
        new = frames[f"new_{slot}"]["prediction"].to_numpy(dtype=np.float64)
        delta += WEIGHTS[slot] * (unit(new) - unit(old))
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
        "slot_weights": WEIGHTS,
        "replacements": {
            slot: {
                "old": str(paths[f"old_{slot}"]),
                "new": str(paths[f"new_{slot}"]),
                "old_vs_new_test_correlation": correlations[slot],
            }
            for slot in WEIGHTS
        },
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
        "submission_policy": "fallback for second daily submission if temporal Transformer is weaker than LB 0.145",
    }
    (METADATA / f"{RUN_NAME}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


