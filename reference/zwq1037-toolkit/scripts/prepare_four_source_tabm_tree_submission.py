"""Prepare the selected four-source TabM/XGBoost fulltrain submission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_NAME = "tabm_four_source_45_20_20_xgb15_fulltrain"
WEIGHTS = {
    "tabm_original_centered": 0.45,
    "tabm_corrprune_raw": 0.20,
    "tabm_cosine_raw": 0.20,
    "xgboost_centered": 0.15,
}


def unit(values: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(values))
    if not norm:
        raise AssertionError("Cannot normalize a zero prediction vector.")
    return values / norm, norm


def centered_unit(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    source_mean = float(values.mean())
    normalized, norm = unit(values - source_mean)
    return normalized, source_mean, norm


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    base = pd.read_feather(
        prediction_dir / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    corrprune = pd.read_feather(
        prediction_dir / "tabm006_corrprune_fulltrain_test.feather"
    )
    cosine_tabm = pd.read_feather(
        prediction_dir / "tabm003_cosine_fulltrain_test.feather"
    )
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    reference_ids = template["sample_id"].to_numpy()
    for name, frame in [
        ("base", base),
        ("corrprune", corrprune),
        ("cosine", cosine_tabm),
    ]:
        if not np.array_equal(frame["sample_id"].to_numpy(), reference_ids):
            raise AssertionError(f"{name} prediction IDs do not match the template.")

    original_unit, original_mean, original_norm = centered_unit(
        base["tabm_prediction"].to_numpy(dtype=np.float64)
    )
    xgboost_unit, xgboost_mean, xgboost_norm = centered_unit(
        base["tree_prediction"].to_numpy(dtype=np.float64)
    )
    corrprune_unit, corrprune_norm = unit(
        corrprune["prediction"].to_numpy(dtype=np.float64)
    )
    cosine_unit, cosine_norm = unit(
        cosine_tabm["prediction"].to_numpy(dtype=np.float64)
    )
    prediction = (
        WEIGHTS["tabm_original_centered"] * original_unit
        + WEIGHTS["tabm_corrprune_raw"] * corrprune_unit
        + WEIGHTS["tabm_cosine_raw"] * cosine_unit
        + WEIGHTS["xgboost_centered"] * xgboost_unit
    )
    if not np.isfinite(prediction).all():
        raise AssertionError("Four-source blend contains non-finite values.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate IDs or missing values.")
    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    submission_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{OUTPUT_NAME}.csv"
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {"sample_id": reference_ids, "prediction": prediction}
    ).to_feather(prediction_path)

    search_result = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-BLEND-008-FOUR-SOURCE-SEARCH"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    selected = search_result["selected_candidate"]
    metadata = {
        "output_name": OUTPUT_NAME,
        "training_months": "0-70 for every source",
        "source_prediction_files": {
            "original_tabm_and_xgboost": str(
                prediction_dir / "tabm001_tree053r_blend75_fulltrain_test.feather"
            ),
            "correlation_pruned_tabm": str(
                prediction_dir / "tabm006_corrprune_fulltrain_test.feather"
            ),
            "cosine_loss_tabm": str(
                prediction_dir / "tabm003_cosine_fulltrain_test.feather"
            ),
        },
        "weights": WEIGHTS,
        "transformation": {
            "tabm_original": "subtract own test mean, then L2 normalize",
            "xgboost": "subtract own test mean, then L2 normalize",
            "tabm_corrprune": "L2 normalize without centering",
            "tabm_cosine": "L2 normalize without centering",
            "uses_target_labels": False,
        },
        "source_statistics": {
            "tabm_original_mean_removed": original_mean,
            "tabm_original_centered_norm": original_norm,
            "xgboost_mean_removed": xgboost_mean,
            "xgboost_centered_norm": xgboost_norm,
            "tabm_corrprune_norm": corrprune_norm,
            "tabm_cosine_norm": cosine_norm,
        },
        "local_validation": {
            key: float(selected[key])
            for key in [
                "overall",
                "no66",
                "recent",
                "early",
                "primary_std",
                "primary_worst",
                "primary_q25",
                "primary_lomo_min",
                "primary_lomo_mean",
            ]
        },
        "test_rows": int(len(submission)),
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
