"""Prepare aggressive and guarded four-source submissions using TabM trim1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = {
    "tabm_trim1_four_source_aggressive_45_20_20_xgb15_fulltrain": {
        "validation_candidate": "aggressive_trim1",
        "weights": {
            "tabm_trim1_centered": 0.45,
            "tabm_corrprune_raw": 0.20,
            "tabm_cosine_raw": 0.20,
            "xgboost_centered": 0.15,
        },
    },
    "tabm_trim1_four_source_guarded_50_25_05_xgb20_fulltrain": {
        "validation_candidate": "guarded_trim1",
        "weights": {
            "tabm_trim1_centered": 0.50,
            "tabm_corrprune_raw": 0.25,
            "tabm_cosine_raw": 0.05,
            "xgboost_centered": 0.20,
        },
    },
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
    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    base = pd.read_feather(
        prediction_dir / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    trimmed = pd.read_feather(
        prediction_dir / "tabm001_member_aggregations_fulltrain_test.feather"
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
        ("trimmed", trimmed),
        ("corrprune", corrprune),
        ("cosine", cosine_tabm),
    ]:
        if not np.array_equal(frame["sample_id"].to_numpy(), reference_ids):
            raise AssertionError(f"{name} test IDs are not aligned.")

    trim1_unit, trim1_mean, trim1_norm = centered_unit(
        trimmed["trim1_prediction"].to_numpy(dtype=np.float64)
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
    sources = {
        "tabm_trim1_centered": trim1_unit,
        "tabm_corrprune_raw": corrprune_unit,
        "tabm_cosine_raw": cosine_unit,
        "xgboost_centered": xgboost_unit,
    }
    validation_results = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-002-BLENDS"
        / "trimmed_blend_results.csv"
    )

    summaries = []
    for output_name, specification in CANDIDATES.items():
        weights = specification["weights"]
        prediction = sum(weights[name] * sources[name] for name in weights)
        if not np.isfinite(prediction).all():
            raise AssertionError(f"{output_name} contains non-finite values.")
        submission = template[["sample_id"]].copy()
        submission["prediction"] = prediction
        if submission["sample_id"].duplicated().any() or submission.isna().any().any():
            raise AssertionError(f"{output_name} contains duplicate IDs or missing values.")
        submission_path = submission_dir / f"{output_name}.csv"
        prediction_path = prediction_dir / f"{output_name}_test.feather"
        metadata_path = metadata_dir / f"{output_name}.json"
        submission.to_csv(submission_path, index=False)
        pd.DataFrame(
            {"sample_id": reference_ids, "prediction": prediction}
        ).to_feather(prediction_path)

        local_row = validation_results.loc[
            validation_results["candidate"] == specification["validation_candidate"]
        ].iloc[0]
        local_metrics = {
            key: (int(local_row[key]) if key == "primary_months_improved" else float(local_row[key]))
            for key in [
                "overall",
                "no66",
                "recent",
                "early",
                "primary_std",
                "primary_worst",
                "primary_q25",
                "primary_months_improved",
                "primary_month_min_change",
                "primary_lomo_min",
                "primary_lomo_mean",
            ]
        }
        metadata = {
            "output_name": output_name,
            "training_months": "0-70 for every trained source",
            "member_aggregation": (
                "sort 16 original TabM member predictions per row, drop one "
                "lowest and one highest, average remaining 14"
            ),
            "aggregation_uses_labels": False,
            "weights": weights,
            "source_statistics": {
                "tabm_trim1_mean_removed": trim1_mean,
                "tabm_trim1_centered_norm": trim1_norm,
                "xgboost_mean_removed": xgboost_mean,
                "xgboost_centered_norm": xgboost_norm,
                "tabm_corrprune_norm": corrprune_norm,
                "tabm_cosine_norm": cosine_norm,
            },
            "local_validation": local_metrics,
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
        summaries.append(metadata)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
