"""Test whether Order20 improves the 359-feature XGBoost on the fixed late fold."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

import exp_tabm_001_exp053r_features as tabm
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER20_COLUMNS
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_027_031_xgboost_capacity_regularization import fit_model, model_parameters


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP-TREE-077-RELATIVE319-XS40-ORDER20"

import sys
sys.path.insert(0, str(PROJECT / "data/interim/kaggle_kernels/relative319_xs_tabm_notebook"))
from run_extracted import TOP_FEATURES, add_relative_features as add_xs_features


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def main() -> None:
    output_dir = PROJECT / "data/interim/tree_experiments" / EXPERIMENT_ID / "train059_valid6270_no66"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, baseline_columns, _, _ = tabm.load_exp053r_data(PROJECT)
    add_relative_features(PROJECT, frame)
    months = frame["month"].to_numpy(dtype=np.int16)
    selected = frame[TOP_FEATURES].to_numpy(dtype=np.float32)
    xs_values, xs_columns = add_xs_features(selected, months, TOP_FEATURES)
    frame = pd.concat(
        [frame, pd.DataFrame(xs_values, columns=xs_columns, index=frame.index)],
        axis=1,
        copy=False,
    )

    order = pd.read_feather(
        PROJECT / "data/processed/train_order_quote_position_features.feather",
        columns=["sample_id", *ORDER20_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(frame["sample_id"].to_numpy(), order["sample_id"].to_numpy()):
        raise AssertionError("Order20 sample IDs do not align with the training frame")
    for column in ORDER20_COLUMNS:
        frame[column] = order[column].to_numpy(dtype=np.float32)
    del order

    feature_columns = [*baseline_columns, *RELATIVE_COLUMNS, *xs_columns, *ORDER20_COLUMNS]
    if len(feature_columns) != 379 or len(set(feature_columns)) != 379:
        raise AssertionError("Expected 379 unique features")

    train_mask = months <= 59
    valid_mask = (months >= 62) & (months <= 70) & (months != 66)
    centered_target = frame.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    parameters = model_parameters(
        n_estimators=800,
        colsample_bytree=0.8,
        device="cuda",
        n_jobs=2,
    )
    model, training_seconds = fit_model(
        frame, feature_columns, train_mask, centered_target, parameters
    )
    ids = frame.loc[valid_mask, "sample_id"].to_numpy()
    target = frame.loc[valid_mask, "target"].to_numpy(dtype=np.float64)
    prediction = np.asarray(
        model.predict(frame.loc[valid_mask, feature_columns]), dtype=np.float64
    )

    baseline = pd.read_feather(
        PROJECT / "data/interim/tree_experiments/EXP-TREE-075-RELATIVE319-XS40/train059_valid6270_no66/validation_predictions.feather",
        columns=["sample_id", "candidate_xs40"],
    )
    if not np.array_equal(ids, baseline["sample_id"].to_numpy()):
        raise AssertionError("Tree359 baseline IDs do not align")
    baseline_prediction = baseline["candidate_xs40"].to_numpy(dtype=np.float64)
    scores = {
        "baseline_tree359": cosine(target, baseline_prediction),
        "candidate_tree379": cosine(target, prediction),
    }
    scores["delta"] = scores["candidate_tree379"] - scores["baseline_tree359"]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "train_months": "0-59",
        "purged_months": "60-61",
        "validation_months": "62-70",
        "excluded_validation_months": [66],
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "feature_count": len(feature_columns),
        "added_features": list(ORDER20_COLUMNS),
        "parameters": parameters,
        "training_seconds": training_seconds,
        "scores": scores,
    }
    pd.DataFrame(
        {
            "sample_id": ids,
            "month": months[valid_mask],
            "target": target,
            "baseline_tree359": baseline_prediction,
            "candidate_tree379": prediction,
        }
    ).to_feather(output_dir / "validation_predictions.feather")
    save_model_safely(model, output_dir / "model.json")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
