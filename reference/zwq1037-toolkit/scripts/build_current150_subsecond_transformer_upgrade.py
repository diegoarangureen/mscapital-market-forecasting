"""Replace the Transformer slot in the proven 0.150 blend with its event hybrid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
RUN_NAME = "current150_subsecond_transformer25_upgrade"
TRANSFORMER_SLOT_WEIGHT = 0.16
COMMON_GROUPS = 38


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, expected_ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if expected_ids is not None and not np.array_equal(
        frame["sample_id"].to_numpy(), expected_ids
    ):
        raise AssertionError(f"IDs do not align: {path}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Nonfinite predictions: {path}")
    return frame


def main() -> None:
    paths = {
        "base": SUBMISSIONS
        / "factorized_full_replace_corr095_current138k32_equal38_common.csv",
        "old_transformer": PROJECT
        / "data/interim/kaggle_outputs/multistream_factorized_transformer_fulltrain"
        / "factorized_transformer/factorized_transformer379_full_e4.csv",
        "new_transformer": SUBMISSIONS
        / "fixed75_transformer25_subsecond_event_full.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    base = load(paths["base"])
    ids = base["sample_id"].to_numpy()
    old = load(paths["old_transformer"], ids)["prediction"].to_numpy(
        dtype=np.float64
    )
    new = load(paths["new_transformer"], ids)["prediction"].to_numpy(
        dtype=np.float64
    )

    # The base blend was assembled from standardized members. Replace only the
    # 16% Transformer slot and preserve its 38-group common component exactly.
    delta = TRANSFORMER_SLOT_WEIGHT * (unit(new) - unit(old))
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
            "new": str(paths["new_transformer"]),
            "new_internal_weights": {
                "old_transformer": 0.75,
                "subsecond_event_model": 0.25,
            },
            "effective_full_blend_weights": {
                "old_transformer": 0.12,
                "subsecond_event_model": 0.04,
            },
        },
        "validation_evidence": {
            "old_transformer_62_70_without_66": 0.1553845147,
            "new_hybrid_62_70_without_66": 0.1590078803,
            "selection_62_65_delta": 0.0047237172,
            "forward_67_70_delta": 0.0031170177,
            "weight_search": False,
        },
        "test_diagnostics": {
            "old_vs_new_transformer_correlation": float(np.corrcoef(old, new)[0, 1]),
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
    METADATA.mkdir(parents=True, exist_ok=True)
    (METADATA / f"{RUN_NAME}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
