"""Evaluate LightGBM on the robust EXP-TREE-037 feature set without recency weights."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import lightgbm
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as TRANSACTION_GAP_FEATURES
from exp_tree_017_019_xgboost_target_and_monthly_transforms import assert_prediction_alignment
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import (
    add_event_features,
    evaluate_and_save,
)
from train_full_xgboost_submissions import load_features


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    transaction_gaps = pd.read_feather(
        processed_dir / "train_transaction_event_gap_features.feather"
    )
    labels = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    model_data = (
        model_data.merge(
            transaction_gaps, on="sample_id", how="left", validate="one_to_one"
        )
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
    )
    del transaction_gaps, labels
    gc.collect()

    exp034_features = base_features + TRANSACTION_FEATURE_COLUMNS + ORDER_MULTI_FEATURE_COLUMNS
    feature_columns = exp034_features + TRANSACTION_GAP_FEATURES
    if len(feature_columns) != 159 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Unexpected EXP-TREE-047 feature schema.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    raw_target = model_data.loc[train_mask, "target"].to_numpy(dtype=np.float64)
    target_mean = float(raw_target.mean())
    centered_target = raw_target - target_mean
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-037_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    parameters = {
        "objective": "regression",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = LGBMRegressor(**parameters)
    start_time = time.perf_counter()
    model.fit(model_data.loc[train_mask, feature_columns], centered_target)
    training_seconds = time.perf_counter() - start_time
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id="EXP-TREE-047",
        description="LightGBM fixed-800 comparison on EXP037's 159 features without recency weights",
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-037",
        save_model=False,
    )
    model_path = project_dir / "outputs" / "models" / "exp-tree-047.joblib"
    joblib.dump(model, model_path)
    metadata["model_family"] = "LightGBM"
    metadata["lightgbm_version"] = lightgbm.__version__
    metadata["training_target_mean"] = target_mean
    metadata["model_path"] = str(model_path)
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-047"
        / "config.json"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
