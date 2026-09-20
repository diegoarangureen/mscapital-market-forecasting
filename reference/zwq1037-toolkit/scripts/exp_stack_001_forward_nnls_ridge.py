"""Fit a non-negative ridge stack on early OOF months and test it forward."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


SOURCE_FILES = {
    "tabm_original_centered": "exp-tabm-001_valid.feather",
    "tabm_corrprune_raw": "exp-tabm-006-corrprune_valid.feather",
    "tabm_cosine_raw": "exp-tabm-003-cosine_valid.feather",
    "xgboost_centered": "exp-tree-053r_valid.feather",
}
RIDGE_RATIOS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
FIXED_BASELINE_WEIGHTS = np.asarray([0.80, 0.00, 0.00, 0.20], dtype=np.float64)
FOUR_SOURCE_WEIGHTS = np.asarray([0.45, 0.20, 0.20, 0.15], dtype=np.float64)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def transform_source(name: str, values: np.ndarray) -> np.ndarray:
    if name.endswith("_centered"):
        values = values - values.mean()
    return values / np.linalg.norm(values)


def fit_nnls_ridge(
    features: np.ndarray, target: np.ndarray, ridge_ratio: float
) -> np.ndarray:
    gram_scale = float(np.trace(features.T @ features) / features.shape[1])
    penalty = np.sqrt(max(ridge_ratio * gram_scale, 0.0))
    if penalty:
        augmented_features = np.vstack(
            [features, penalty * np.eye(features.shape[1], dtype=np.float64)]
        )
        augmented_target = np.concatenate(
            [target, np.zeros(features.shape[1], dtype=np.float64)]
        )
    else:
        augmented_features = features
        augmented_target = target
    weights, _ = nnls(augmented_features, augmented_target)
    if not weights.sum():
        raise AssertionError("NNLS returned an all-zero weight vector.")
    return weights / weights.sum()


def evaluate_windows(
    target: np.ndarray, prediction: np.ndarray, months: np.ndarray
) -> dict[str, float]:
    month_values = np.sort(np.unique(months))
    monthly = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    primary = monthly[(month_values >= 62) & (month_values != 66)]
    return {
        "overall_60_70": cosine(target, prediction),
        "early_60_64": cosine(target[months <= 64], prediction[months <= 64]),
        "selection_63_64": cosine(
            target[(months >= 63) & (months <= 64)],
            prediction[(months >= 63) & (months <= 64)],
        ),
        "no66_62_70": cosine(
            target[(months >= 62) & (months != 66)],
            prediction[(months >= 62) & (months != 66)],
        ),
        "forward_67_70": cosine(target[months >= 67], prediction[months >= 67]),
        "primary_std": float(primary.std(ddof=0)),
        "primary_worst": float(primary.min()),
        "primary_q25": float(np.quantile(primary, 0.25)),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_original_centered"]
    validation_rows = reference[["sample_id", "month", "target"]]
    columns = []
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        columns.append(
            transform_source(name, frame["prediction"].to_numpy(dtype=np.float64))
        )
    features = np.column_stack(columns)
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()

    meta_train = (months >= 60) & (months <= 62)
    meta_select = (months >= 63) & (months <= 64)
    selection_rows = []
    for ridge_ratio in RIDGE_RATIOS:
        weights = fit_nnls_ridge(features[meta_train], target[meta_train], ridge_ratio)
        prediction = features @ weights
        selection_rows.append(
            {
                "ridge_ratio": ridge_ratio,
                **{
                    f"weight_{name}": float(weight)
                    for name, weight in zip(SOURCE_FILES, weights, strict=True)
                },
                "train_60_62_cosine": cosine(
                    target[meta_train], prediction[meta_train]
                ),
                "selection_63_64_cosine": cosine(
                    target[meta_select], prediction[meta_select]
                ),
                "selection_month63": cosine(
                    target[months == 63], prediction[months == 63]
                ),
                "selection_month64": cosine(
                    target[months == 64], prediction[months == 64]
                ),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["selection_63_64_cosine", "selection_month64"], ascending=False
    )
    selected_ridge_ratio = float(selection.iloc[0]["ridge_ratio"])

    refit_mask = (months >= 60) & (months <= 64)
    learned_weights = fit_nnls_ridge(
        features[refit_mask], target[refit_mask], selected_ridge_ratio
    )
    candidates = {
        "centered_80_20_baseline": FIXED_BASELINE_WEIGHTS,
        "four_source_fixed": FOUR_SOURCE_WEIGHTS,
        "forward_nnls_ridge": learned_weights,
    }
    evaluation_rows = []
    for name, weights in candidates.items():
        prediction = features @ weights
        evaluation_rows.append(
            {
                "candidate": name,
                **{
                    f"weight_{source}": float(weight)
                    for source, weight in zip(SOURCE_FILES, weights, strict=True)
                },
                **evaluate_windows(target, prediction, months),
            }
        )
    evaluation = pd.DataFrame(evaluation_rows)

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-STACK-001-FORWARD-NNLS-RIDGE"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(run_dir / "ridge_selection_60_62_to_63_64.csv", index=False)
    evaluation.to_csv(run_dir / "forward_evaluation.csv", index=False)
    metadata = {
        "sources": SOURCE_FILES,
        "meta_train_months": "60-62",
        "ridge_selection_months": "63-64",
        "gap_months_not_used": [65, 66],
        "forward_evaluation_months": "67-70",
        "ridge_ratios": RIDGE_RATIOS,
        "selected_ridge_ratio": selected_ridge_ratio,
        "refit_months": "60-64",
        "learned_weights": {
            name: float(weight)
            for name, weight in zip(SOURCE_FILES, learned_weights, strict=True)
        },
        "uses_forward_labels_for_model_selection": False,
        "result": evaluation.to_dict(orient="records"),
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(selection.to_string(index=False), flush=True)
    print(evaluation.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
