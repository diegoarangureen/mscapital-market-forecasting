"""Add 20% validated XS40 TabM to the current Public LB 0.138 recipe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_014_quantile_stage_curves import unit


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_NAME = "current138_xs40_tabm20"


def main() -> None:
    submissions = PROJECT_DIR / "outputs" / "submissions"
    current = pd.read_csv(
        submissions / "current135_hybrid2_conv1_transformer30.csv"
    )
    xs40 = pd.read_csv(
        submissions / "tabm_relative319_xs40_seed42_fulltrain.csv"
    )
    if not np.array_equal(
        current["sample_id"].to_numpy(),
        xs40["sample_id"].to_numpy(),
    ):
        raise AssertionError("Submission IDs differ.")
    prediction = (
        0.80 * unit(current["prediction"].to_numpy(dtype=np.float64))
        + 0.20 * unit(xs40["prediction"].to_numpy(dtype=np.float64))
    )
    if not np.isfinite(prediction).all():
        raise AssertionError("Nonfinite submission.")
    path = submissions / f"{RUN_NAME}.csv"
    pd.DataFrame({
        "sample_id": current["sample_id"],
        "prediction": prediction,
    }).to_csv(path, index=False)
    metadata = {
        "run_name": RUN_NAME,
        "source_weights": {
            "current138": 0.80,
            "xs40_tabm": 0.20,
        },
        "local_scores": {
            "50_59": 0.156898,
            "62_70_no66": 0.157651,
        },
        "local_deltas_vs_current138": {
            "50_59": 0.000576,
            "62_70_no66": 0.001200,
        },
        "rows": len(prediction),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "submission_status": "prepared_not_uploaded",
        "output_path": str(path),
    }
    out = (
        PROJECT_DIR / "outputs" / "submission_metadata"
        / f"{RUN_NAME}.json"
    )
    out.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()