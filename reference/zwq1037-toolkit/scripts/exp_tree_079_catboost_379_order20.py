"""Apply the proven GPU CatBoost recipe to the latest 379 features."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import exp_tabm_001_exp053r_features as tabm
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER20_COLUMNS
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP-TREE-079-CATBOOST-379"
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
    frame = pd.concat([frame, pd.DataFrame(xs_values, columns=xs_columns, index=frame.index)], axis=1, copy=False)
    order = pd.read_feather(
        PROJECT / "data/processed/train_order_quote_position_features.feather",
        columns=["sample_id", *ORDER20_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(frame["sample_id"].to_numpy(), order["sample_id"].to_numpy()):
        raise AssertionError("Order20 sample IDs do not align")
    for column in ORDER20_COLUMNS:
        frame[column] = order[column].to_numpy(dtype=np.float32)
    del order
    feature_columns = [*baseline_columns, *RELATIVE_COLUMNS, *xs_columns, *ORDER20_COLUMNS]
    if len(feature_columns) != 379 or len(set(feature_columns)) != 379:
        raise AssertionError("Expected 379 unique features")

    train_mask = months <= 59
    valid_mask = (months >= 62) & (months <= 70) & (months != 66)
    target = frame["target"].to_numpy(dtype=np.float64)
    centered_target = target[train_mask] - float(target[train_mask].mean())
    parameters = {
        "loss_function": "RMSE",
        "iterations": 800,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 5.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_strength": 0.5,
        "border_count": 128,
        "random_seed": 42,
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.80,
        "thread_count": 2,
        "allow_writing_files": False,
        "verbose": 100,
    }
    model = CatBoostRegressor(**parameters)
    started = time.perf_counter()
    model.fit(frame.loc[train_mask, feature_columns], centered_target)
    training_seconds = time.perf_counter() - started
    prediction = np.asarray(model.predict(frame.loc[valid_mask, feature_columns]), dtype=np.float64)
    ids = frame.loc[valid_mask, "sample_id"].to_numpy()
    valid_target = target[valid_mask]

    baseline = pd.read_feather(
        PROJECT / "data/interim/tree_experiments/EXP-TREE-077-RELATIVE319-XS40-ORDER20/train059_valid6270_no66/validation_predictions.feather",
        columns=["sample_id", "candidate_tree379"],
    )
    if not np.array_equal(ids, baseline["sample_id"].to_numpy()):
        raise AssertionError("XGBoost379 baseline IDs do not align")
    baseline_prediction = baseline["candidate_tree379"].to_numpy(dtype=np.float64)
    scores = {
        "baseline_xgboost379": cosine(valid_target, baseline_prediction),
        "candidate_catboost379": cosine(valid_target, prediction),
    }
    scores["delta"] = scores["candidate_catboost379"] - scores["baseline_xgboost379"]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model_family": "CatBoost",
        "catboost_version": catboost.__version__,
        "train_months": "0-59",
        "purged_months": "60-61",
        "validation_months": "62-70",
        "excluded_validation_months": [66],
        "feature_count": len(feature_columns),
        "parameters": parameters,
        "training_seconds": training_seconds,
        "scores": scores,
    }
    pd.DataFrame({
        "sample_id": ids,
        "month": months[valid_mask],
        "target": valid_target,
        "baseline_xgboost379": baseline_prediction,
        "candidate_catboost379": prediction,
    }).to_feather(output_dir / "validation_predictions.feather")
    model.save_model(output_dir / "model.cbm")
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
