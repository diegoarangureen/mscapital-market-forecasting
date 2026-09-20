"""Prepare predeclared trimmed six-source blend candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = {
    "tabm_double_trim_blend005_fulltrain": {
        "original": 0.35,
        "corrprune": 0.20,
        "cosine": 0.15,
        "xgboost": 0.10,
        "lightgbm": 0.10,
        "histgb": 0.10,
    },
    "tabm_double_trim_six_robust_40_20_20_05_05_10_fulltrain": {
        "original": 0.40,
        "corrprune": 0.20,
        "cosine": 0.20,
        "xgboost": 0.05,
        "lightgbm": 0.05,
        "histgb": 0.10,
    },
    "tabm_double_trim_nohist_40_20_20_xgb05_lgbm15_fulltrain": {
        "original": 0.40,
        "corrprune": 0.20,
        "cosine": 0.20,
        "xgboost": 0.05,
        "lightgbm": 0.15,
        "histgb": 0.00,
    },
    "tabm_double_trim_six_overall_55_20_10_05_05_05_fulltrain": {
        "original": 0.55,
        "corrprune": 0.20,
        "cosine": 0.10,
        "xgboost": 0.05,
        "lightgbm": 0.05,
        "histgb": 0.05,
    },
}


def unit(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    norm = float(np.linalg.norm(values))
    if not norm:
        raise AssertionError("Cannot normalize a zero vector.")
    return values / norm, {"mean_removed": 0.0, "norm": norm}


def centered_unit(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mean = float(values.mean())
    normalized, statistics = unit(values - mean)
    statistics["mean_removed"] = mean
    return normalized, statistics


def find_local_row(results: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    mask = np.ones(len(results), dtype=bool)
    for name, value in weights.items():
        mask &= np.isclose(results[f"weight_{name}"].to_numpy(), value)
    rows = results.loc[mask]
    if len(rows) != 1:
        raise AssertionError(f"Expected one local row, got {len(rows)} for {weights}.")
    return rows.iloc[0]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    ids = template["sample_id"].to_numpy()

    frames = {
        "base": pd.read_feather(
            prediction_dir / "tabm001_tree053r_blend75_fulltrain_test.feather"
        ),
        "original": pd.read_feather(
            prediction_dir / "tabm001_member_aggregations_fulltrain_test.feather"
        ),
        "corrprune": pd.read_feather(
            prediction_dir / "tabm006_corrprune_fulltrain_test.feather"
        ),
        "cosine": pd.read_feather(
            prediction_dir / "tabm003_cosine_member_aggregations_fulltrain_test.feather"
        ),
        "lightgbm": pd.read_feather(
            prediction_dir / "lightgbm068_fulltrain_test.feather"
        ),
        "histgb": pd.read_feather(
            prediction_dir / "histgb007_fulltrain_test.feather"
        ),
    }
    for name, frame in frames.items():
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError(f"{name} IDs are not aligned to the template.")

    source_values = {
        "original": frames["original"]["trim1_prediction"].to_numpy(dtype=np.float64),
        "corrprune": frames["corrprune"]["prediction"].to_numpy(dtype=np.float64),
        "cosine": frames["cosine"]["trim1_prediction"].to_numpy(dtype=np.float64),
        "xgboost": frames["base"]["tree_prediction"].to_numpy(dtype=np.float64),
        "lightgbm": frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64),
        "histgb": frames["histgb"]["prediction"].to_numpy(dtype=np.float64),
    }
    centered_names = {"original", "xgboost", "lightgbm"}
    normalized: dict[str, np.ndarray] = {}
    source_statistics: dict[str, dict[str, float]] = {}
    for name, values in source_values.items():
        transformation = centered_unit if name in centered_names else unit
        normalized[name], source_statistics[name] = transformation(values)

    local_results = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-011-TRIMMED-SIX-SOURCE"
        / "all_candidates.csv"
    )
    metric_names = [
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
    summaries: list[dict[str, object]] = []
    for output_name, weights in CANDIDATES.items():
        prediction = sum(weights[name] * normalized[name] for name in weights)
        submission = template[["sample_id"]].copy()
        submission["prediction"] = prediction
        if (
            not np.isfinite(prediction).all()
            or submission["sample_id"].duplicated().any()
            or submission.isna().any().any()
        ):
            raise AssertionError(f"Invalid submission values for {output_name}.")
        local_row = find_local_row(local_results, weights)
        local_metrics = {
            name: (
                int(local_row[name])
                if name == "primary_months_improved"
                else float(local_row[name])
            )
            for name in metric_names
        }

        submission_path = submission_dir / f"{output_name}.csv"
        prediction_path = prediction_dir / f"{output_name}_test.feather"
        metadata_path = metadata_dir / f"{output_name}.json"
        submission.to_csv(submission_path, index=False)
        submission.to_feather(prediction_path)
        metadata = {
            "output_name": output_name,
            "training_months": "0-70 for every trained source",
            "aggregation_uses_labels": False,
            "member_aggregation": {
                "tabm_original": "trim1",
                "tabm_corrprune": "mean",
                "tabm_cosine": "trim1",
            },
            "weights": weights,
            "source_transformations": {
                name: ("center_then_l2" if name in centered_names else "l2")
                for name in weights
            },
            "source_statistics": source_statistics,
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
