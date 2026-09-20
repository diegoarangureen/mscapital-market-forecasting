"""Prepare the selected centered-component 80/20 TabM/XGBoost submission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_NAME = "tabm001_tree053r_centered_components_blend80_fulltrain"
TABM_WEIGHT = 0.80
TREE_WEIGHT = 0.20


def centered_unit_vector(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Subtract the vector's own mean and return a unit-L2 vector."""
    source_mean = float(values.mean())
    centered = values - source_mean
    centered_norm = float(np.linalg.norm(centered))
    if not centered_norm:
        raise AssertionError("Cannot normalize a zero prediction vector.")
    return centered / centered_norm, source_mean, centered_norm


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    source_path = (
        project_dir
        / "outputs"
        / "predictions"
        / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    source = pd.read_feather(source_path)
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if not np.array_equal(source["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Source predictions do not match submission template order.")

    tabm_raw = source["tabm_prediction"].to_numpy(dtype=np.float64)
    tree_raw = source["tree_prediction"].to_numpy(dtype=np.float64)
    tabm_unit, tabm_mean, tabm_centered_norm = centered_unit_vector(tabm_raw)
    tree_unit, tree_mean, tree_centered_norm = centered_unit_vector(tree_raw)
    prediction = TABM_WEIGHT * tabm_unit + TREE_WEIGHT * tree_unit
    if not np.isfinite(prediction).all():
        raise AssertionError("Blended prediction contains non-finite values.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
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
            "prediction": prediction,
        }
    ).to_feather(prediction_path)

    audit = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-POST-002-COMPONENT-CENTERING"
        / "centered_component_weight_grid.csv"
    )
    local_row = audit.loc[np.isclose(audit["tabm_weight"], TABM_WEIGHT)].iloc[0]
    metadata = {
        "output_name": OUTPUT_NAME,
        "source_prediction_file": str(source_path),
        "source_public_score": 0.131,
        "transformation": (
            "subtract each component's own test mean, L2-normalize each centered "
            "component, then blend 80% TabM and 20% XGBoost"
        ),
        "uses_target_labels": False,
        "weights": {"tabm": TABM_WEIGHT, "xgboost": TREE_WEIGHT},
        "source_statistics": {
            "tabm_mean_removed": tabm_mean,
            "tabm_centered_l2_norm": tabm_centered_norm,
            "xgboost_mean_removed": tree_mean,
            "xgboost_centered_l2_norm": tree_centered_norm,
        },
        "local_validation": {
            key: float(local_row[key])
            for key in [
                "overall_cosine",
                "cosine_62_70_without_66",
                "cosine_67_70",
                "cosine_60_64",
                "primary_monthly_std",
                "primary_monthly_worst",
                "primary_monthly_q25",
            ]
        },
        "prediction_summary": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std(ddof=0)),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "l2_norm": float(np.linalg.norm(prediction)),
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
