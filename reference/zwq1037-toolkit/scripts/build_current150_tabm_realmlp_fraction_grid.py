"""Build a small fraction grid for the proven TabM and RealMLP upgrades."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
TABM_SLOT_WEIGHT = 0.15
REALMLP_SLOT_WEIGHT = 0.05
TABM_FRACTIONS = (1.0, 1.25, 1.5)
REALMLP_FRACTIONS = (0.5, 1.0)


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(name: str, ids: np.ndarray | None = None) -> pd.DataFrame:
    path = SUBMISSIONS / name
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"] or len(frame) != 647_896:
        raise AssertionError(f"Invalid submission: {path}")
    if ids is not None and not np.array_equal(frame["sample_id"].to_numpy(), ids):
        raise AssertionError(f"IDs do not align: {path}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Nonfinite prediction: {path}")
    return frame


def token(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    base = load("factorized_full_replace_corr095_current138k32_equal38_common.csv")
    ids = base["sample_id"].to_numpy()
    old_tabm = load("current138_refreshed_tabm385_k32.csv", ids)
    new_tabm = load("current138_refreshed_tabm385_temporal3x3.csv", ids)
    old_realmlp = load("standalone_realmlp379_corr095.csv", ids)
    new_realmlp = load("realmlp379_rq16_temporal3fold_e8_fold_end57.csv", ids)
    tabm_delta = TABM_SLOT_WEIGHT * (
        unit(new_tabm["prediction"]) - unit(old_tabm["prediction"])
    )
    realmlp_delta = REALMLP_SLOT_WEIGHT * (
        unit(new_realmlp["prediction"]) - unit(old_realmlp["prediction"])
    )
    groups = (np.arange(len(ids), dtype=np.int64) * 38 // len(ids)).astype(np.int16)
    outputs = []
    for tabm_fraction in TABM_FRACTIONS:
        for realmlp_fraction in REALMLP_FRACTIONS:
            delta = tabm_fraction * tabm_delta + realmlp_fraction * realmlp_delta
            for group in np.unique(groups):
                mask = groups == group
                delta[mask] -= delta[mask].mean()
            prediction = base["prediction"].to_numpy(dtype=np.float64) + delta
            run_name = (
                "current150_tabmfrac"
                + token(tabm_fraction)
                + "_realmlpfrac"
                + token(realmlp_fraction)
            )
            path = SUBMISSIONS / f"{run_name}.csv"
            pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(path, index=False)
            outputs.append(
                {
                    "run_name": run_name,
                    "tabm_fraction": tabm_fraction,
                    "realmlp_fraction": realmlp_fraction,
                    "effective_tabm_slot_weight": TABM_SLOT_WEIGHT * tabm_fraction,
                    "effective_realmlp_slot_weight": REALMLP_SLOT_WEIGHT * realmlp_fraction,
                    "correlation_with_current_best": float(
                        np.corrcoef(base["prediction"], prediction)[0, 1]
                    ),
                    "rows": len(ids),
                    "finite": bool(np.isfinite(prediction).all()),
                    "output_path": str(path),
                }
            )
    report = {
        "run_name": "current150_tabm_realmlp_fraction_grid",
        "base_public_lb": 0.150,
        "tabm_evidence": "standalone public LB 0.134 -> 0.140",
        "realmlp_evidence": "late local cosine 0.154668 -> 0.155592",
        "outputs": outputs,
        "submission_status": "diagnostic_candidates_not_submitted",
    }
    (METADATA / "current150_tabm_realmlp_fraction_grid.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
