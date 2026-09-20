"""Prepare, but do not upload, a globally centered public-best submission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_NAME = "tabm001_tree053r_blend75_fulltrain_centered"


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    source_path = (
        project_dir
        / "outputs"
        / "predictions"
        / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    template_path = project_dir / "data" / "raw" / "submission.csv"
    source = pd.read_feather(source_path)
    template = pd.read_csv(template_path)
    if not np.array_equal(source["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Source predictions do not match submission template order.")

    raw_prediction = source["prediction"].to_numpy(dtype=np.float64)
    source_mean = float(raw_prediction.mean())
    centered_prediction = raw_prediction - source_mean
    if not np.isfinite(centered_prediction).all():
        raise AssertionError("Centered prediction contains non-finite values.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = centered_prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate IDs or missing values.")

    submission_dir = project_dir / "outputs" / "submissions"
    prediction_dir = project_dir / "outputs" / "predictions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    for directory in (submission_dir, prediction_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{OUTPUT_NAME}.csv"
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "sample_id": source["sample_id"].to_numpy(),
            "raw_prediction": raw_prediction,
            "prediction": centered_prediction,
        }
    ).to_feather(prediction_path)

    centering_audit = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-POST-001-CENTERING-AUDIT"
        / "global_centering_results.csv"
    )
    local_row = centering_audit.loc[
        centering_audit["model"] == "BLEND001_PUBLIC_BEST"
    ].iloc[0]
    metadata = {
        "output_name": OUTPUT_NAME,
        "source_submission": "tabm001_tree053r_blend75_fulltrain.csv",
        "source_public_score": 0.131,
        "transformation": "prediction = raw_prediction - mean(raw_prediction)",
        "uses_target_labels": False,
        "test_prediction_mean_removed": source_mean,
        "test_rows": int(len(submission)),
        "local_validation": {
            "raw_overall": float(local_row["raw_overall"]),
            "centered_overall": float(local_row["centered_overall"]),
            "raw_no66": float(local_row["raw_no66"]),
            "centered_no66": float(local_row["centered_no66"]),
            "raw_recent": float(local_row["raw_recent"]),
            "centered_recent": float(local_row["centered_recent"]),
            "raw_primary_worst": float(local_row["raw_primary_worst"]),
            "centered_primary_worst": float(local_row["centered_primary_worst"]),
        },
        "prediction_summary": {
            "mean": float(centered_prediction.mean()),
            "std": float(centered_prediction.std(ddof=0)),
            "min": float(centered_prediction.min()),
            "max": float(centered_prediction.max()),
            "l2_norm": float(np.linalg.norm(centered_prediction)),
        },
        "submission_path": str(submission_path),
        "prediction_path": str(prediction_path),
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
