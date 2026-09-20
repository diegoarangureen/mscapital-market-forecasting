"""Compare Relative319 cosine-loss TabM with the fixed MSE baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import cosine


BASELINE_ID = "EXP-TABM-016-RELATIVE-SCALE-ADD12"
CANDIDATE_ID = "EXP-TABM-021-RELATIVE319-COSINE-LOSS"
FOLD = "train059_valid6070"
COMPONENTS = ("tabm_mean", "tabm_trim1", "b001_trim1")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    root = project_dir / "data" / "interim" / "tree_experiments"
    baseline = pd.read_feather(
        root / BASELINE_ID / FOLD / "validation_predictions.feather"
    )
    candidate = pd.read_feather(
        root / CANDIDATE_ID / FOLD / "validation_predictions.feather"
    )
    if not np.array_equal(baseline["sample_id"], candidate["sample_id"]):
        raise AssertionError("MSE and cosine prediction sample IDs differ.")
    if not np.array_equal(baseline["month"], candidate["month"]):
        raise AssertionError("MSE and cosine prediction months differ.")
    if not np.allclose(baseline["target"], candidate["target"]):
        raise AssertionError("MSE and cosine targets differ.")

    months = baseline["month"].to_numpy()
    targets = baseline["target"].to_numpy(dtype=np.float64)
    windows = {
        "60_70": (months >= 60) & (months <= 70),
        "62_70_two_month_gap": (months >= 62) & (months <= 70),
        "62_70_without_66": (months >= 62) & (months <= 70) & (months != 66),
        "67_70": (months >= 67) & (months <= 70),
    }
    result = {
        "experiment_id": CANDIDATE_ID,
        "baseline": BASELINE_ID,
        "train_months": "0-59",
        "windows": {},
    }
    for name, mask in windows.items():
        comparisons = {}
        for component in COMPONENTS:
            mse_prediction = baseline[component].to_numpy(dtype=np.float64)
            cosine_prediction = candidate[component].to_numpy(dtype=np.float64)
            mse_score = cosine(targets[mask], mse_prediction[mask])
            candidate_score = cosine(targets[mask], cosine_prediction[mask])
            comparisons[component] = {
                "mse_baseline": mse_score,
                "cosine_loss": candidate_score,
                "delta": candidate_score - mse_score,
            }
        result["windows"][name] = {
            "rows": int(mask.sum()),
            "components": comparisons,
        }
    output_path = root / CANDIDATE_ID / "confirmation_comparison.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
