"""Replace the owned full-train TabM in the current 0.150 blend with temporal 3x3 TabM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "current150_replace_owned_tabm_temporal3x3"
CURRENT_SLOT_WEIGHT = 0.15


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, reference_ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if reference_ids is not None and not np.array_equal(
        frame["sample_id"].to_numpy(), reference_ids
    ):
        raise AssertionError(f"IDs do not align: {path}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Nonfinite prediction: {path}")
    return frame


def main() -> None:
    submissions = PROJECT / "outputs" / "submissions"
    metadata = PROJECT / "outputs" / "submission_metadata"
    components = pd.read_feather(
        PROJECT / "outputs" / "predictions" / "gru_embedding_seed_ensemble_fulltrain_test.feather"
    )
    old_current = load(submissions / "gru_embedding_seed_ensemble_fulltrain.csv")
    ids = old_current["sample_id"].to_numpy()
    old_tabm = components["tabm_prediction"].to_numpy(dtype=np.float64)
    temporal_tabm_frame = load(
        submissions / "tabm385_k32_temporal3fold_3seed.csv", ids
    )
    temporal_tabm = temporal_tabm_frame["prediction"].to_numpy(dtype=np.float64)

    # Rebuild the current138 member with the same internal weights used for k32.
    refreshed_inner = unit(old_current["prediction"]) + 0.675 * (
        unit(temporal_tabm) - unit(old_tabm)
    )
    temporal_current = 0.80 * unit(refreshed_inner) + 0.20 * unit(temporal_tabm)
    temporal_current_path = submissions / "current138_refreshed_tabm385_temporal3x3.csv"
    pd.DataFrame(
        {"sample_id": ids, "prediction": temporal_current}
    ).to_csv(temporal_current_path, index=False)

    best = load(
        submissions / "factorized_full_replace_corr095_current138k32_equal38_common.csv",
        ids,
    )
    k32_current = load(submissions / "current138_refreshed_tabm385_k32.csv", ids)
    delta = CURRENT_SLOT_WEIGHT * (
        unit(temporal_current) - unit(k32_current["prediction"])
    )
    # Preserve the existing equal-38 common component convention.
    groups = (np.arange(len(ids), dtype=np.int64) * 38 // len(ids)).astype(np.int16)
    for group in np.unique(groups):
        mask = groups == group
        delta[mask] -= delta[mask].mean()
    prediction = best["prediction"].to_numpy(dtype=np.float64) + delta
    output_path = submissions / f"{RUN_NAME}.csv"
    pd.DataFrame(
        {"sample_id": ids, "prediction": prediction}
    ).to_csv(output_path, index=False)

    report = {
        "run_name": RUN_NAME,
        "base": {
            "path": str(submissions / "factorized_full_replace_corr095_current138k32_equal38_common.csv"),
            "public_lb": 0.150,
        },
        "replacement": {
            "current_slot_weight": CURRENT_SLOT_WEIGHT,
            "old_owned_tabm": "TabM385-k32 fulltrain seed42",
            "old_owned_tabm_public_lb": 0.134,
            "new_owned_tabm": "TabM385-k32 temporal folds 52/57/62 x seeds 42/137/2026",
            "new_owned_tabm_public_lb": 0.140,
            "old_vs_new_tabm_test_correlation": float(
                np.corrcoef(old_tabm, temporal_tabm)[0, 1]
            ),
            "old_vs_new_current_member_correlation": float(
                np.corrcoef(k32_current["prediction"], temporal_current)[0, 1]
            ),
        },
        "candidate": {
            "rows": len(ids),
            "finite": bool(np.isfinite(prediction).all()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "correlation_with_current_best": float(
                np.corrcoef(best["prediction"], prediction)[0, 1]
            ),
        },
        "output_path": str(output_path),
        "submission_status": "prepared_not_submitted",
        "submission_policy": "reserve as final blend candidate after standalone calibration",
    }
    (metadata / f"{RUN_NAME}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
