"""Replace the 15% current138 member in the best 0.150 blend with current138-k32."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "factorized_full_replace_corr095_current138k32_equal38_common"
CURRENT_WEIGHT = 0.15
BASE_PUBLIC_SCORE = 0.150
BASE_SUBMISSION_REF = 56250034


def zunit(values: np.ndarray) -> np.ndarray:
    """标准化为零均值、单位方差。 Standardize to zero mean and unit variance."""
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def main() -> None:
    base_path = PROJECT / "outputs/submissions/factorized_full_replace_corr095_realmlp_equal38_common.csv"
    old_current_path = PROJECT / "outputs/submissions/current138_xs40_tabm20.csv"
    new_current_path = PROJECT / "outputs/submissions/current138_refreshed_tabm385_k32.csv"

    base = pd.read_csv(base_path)
    old_current = pd.read_csv(old_current_path)
    new_current = pd.read_csv(new_current_path)

    for name, frame in {
        "base": base,
        "old_current138": old_current,
        "new_current138_k32": new_current,
    }.items():
        assert list(frame.columns) == ["sample_id", "prediction"], name
        assert np.array_equal(frame["sample_id"].to_numpy(), base["sample_id"].to_numpy()), name
        assert np.isfinite(frame["prediction"].to_numpy(float)).all(), name

    # 先替换原融合中 15% 的 current138，再按每月去除替换产生的均值变化。
    # Replace the 15% current138 slot, then remove the group-mean shift caused by the swap.
    raw_delta = CURRENT_WEIGHT * (
        zunit(new_current["prediction"].to_numpy(float))
        - zunit(old_current["prediction"].to_numpy(float))
    )
    row_count = len(base)
    groups = (np.arange(row_count, dtype=np.int64) * 38 // row_count).astype(np.int16)
    centered_delta = raw_delta.copy()
    for group in np.unique(groups):
        mask = groups == group
        centered_delta[mask] -= centered_delta[mask].mean()

    prediction = base["prediction"].to_numpy(float) + centered_delta
    output_path = PROJECT / f"outputs/submissions/{RUN_NAME}.csv"
    pd.DataFrame({
        "sample_id": base["sample_id"].to_numpy(),
        "prediction": prediction,
    }).to_csv(output_path, index=False)

    report = {
        "run_name": RUN_NAME,
        "base_submission": str(base_path),
        "base_kaggle_submission_ref": BASE_SUBMISSION_REF,
        "base_public_score": BASE_PUBLIC_SCORE,
        "replacement": {
            "weight": CURRENT_WEIGHT,
            "old_member": str(old_current_path),
            "new_member": str(new_current_path),
            "new_member_local_62_70_no66": 0.1602189723683853,
            "old_member_local_62_70_no66_recorded": 0.157651,
        },
        "preserved_components": {
            "gpu_tabm142": 0.20,
            "tpu_tabm142": 0.12,
            "yangq_blend142": 0.32,
            "own_realmlp379_corr095": 0.05,
            "factorized_transformer": 0.16,
            "equal38_common_component": True,
        },
        "correlation": {
            "old_vs_new_current138": float(np.corrcoef(
                old_current["prediction"].to_numpy(float),
                new_current["prediction"].to_numpy(float),
            )[0, 1]),
            "base_vs_candidate": float(np.corrcoef(
                base["prediction"].to_numpy(float), prediction
            )[0, 1]),
        },
        "prediction": {
            "rows": int(row_count),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "finite": bool(np.isfinite(prediction).all()),
        },
        "output_path": str(output_path),
        "submission_status": "prepared_not_submitted",
    }
    metadata_path = PROJECT / f"outputs/submission_metadata/{RUN_NAME}.json"
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
