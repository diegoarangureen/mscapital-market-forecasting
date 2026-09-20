"""Average the two validated full-train order20 TabM predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_NAME = "tabm_relative319_xs40_order20_2seed_fulltrain"
SEEDS = (42, 137)


def main() -> None:
    frames = [
        pd.read_csv(
            PROJECT_DIR
            / "outputs"
            / "submissions"
            / f"tabm_relative319_xs40_order20_seed{seed}_fulltrain.csv"
        )
        for seed in SEEDS
    ]
    template = pd.read_csv(PROJECT_DIR / "data" / "raw" / "submission.csv")
    sample_ids = template["sample_id"].to_numpy()
    for seed, frame in zip(SEEDS, frames):
        if list(frame.columns) != ["sample_id", "prediction"]:
            raise AssertionError(f"Unexpected prediction columns for seed {seed}.")
        if not np.array_equal(sample_ids, frame["sample_id"].to_numpy()):
            raise AssertionError(f"Prediction IDs differ from template for seed {seed}.")

    seed_predictions = [frame["prediction"].to_numpy(dtype=np.float64) for frame in frames]
    prediction = (seed_predictions[0] + seed_predictions[1]) / 2.0
    if not np.isfinite(prediction).all():
        raise AssertionError("Nonfinite two-seed prediction.")

    output = pd.DataFrame({"sample_id": sample_ids, "prediction": prediction})
    submission_path = PROJECT_DIR / "outputs" / "submissions" / f"{RUN_NAME}.csv"
    prediction_path = PROJECT_DIR / "outputs" / "predictions" / f"{RUN_NAME}_test.feather"
    output.to_csv(submission_path, index=False)
    output.to_feather(prediction_path)

    validation = json.loads(
        (
            PROJECT_DIR
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-027-ORDER-QUOTE-POSITION20-SEED137"
            / "combined_summary.json"
        ).read_text(encoding="utf-8")
    )
    metadata = {
        "run_name": RUN_NAME,
        "seeds": list(SEEDS),
        "feature_count": 379,
        "train_months": "0-70",
        "test_rows": len(output),
        "validation_2seed": validation,
        "test_prediction_correlation": float(np.corrcoef(*seed_predictions)[0, 1]),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "submission_path": str(submission_path),
        "prediction_path": str(prediction_path),
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path = PROJECT_DIR / "outputs" / "submission_metadata" / f"{RUN_NAME}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
