"""Evaluate fixed-budget LightGBM on the complete EXP053R feature schema."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import (
    add_event_features,
    evaluate_and_save,
)
from exp_tree_052_053_xgboost_gpu_last15 import finalize_metadata
from train_full_xgboost_exp051_submission import (
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    public_features, dropped_public_features = selected_public_features(project_dir)
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    gap = pd.read_feather(processed_dir / "train_transaction_event_gap_features.feather")
    last15 = pd.read_feather(
        processed_dir / "train_market_last15_baseline_features.feather"
    )
    public = load_public_table(project_dir, "train", public_features).rename(
        columns={"target": "public_target"}
    )
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    model_data = (
        model_data.merge(gap, on="sample_id", how="left", validate="one_to_one")
        .merge(public, on="sample_id", how="left", validate="one_to_one")
        .merge(last15, on="sample_id", how="left", validate="one_to_one")
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    target_difference = np.max(
        np.abs(
            model_data["public_target"].to_numpy(dtype=np.float64)
            - model_data["target"].to_numpy(dtype=np.float64)
        )
    )
    if target_difference > 1e-5:
        raise AssertionError("Public and official targets differ.")
    model_data = model_data.drop(columns="public_target")
    model_data["x_rv_15_over_full"] = model_data["m_rv_15"] / (
        model_data["m_rv"] + 1e-8
    )
    del gap, last15, public, labels
    gc.collect()

    feature_columns = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
        + LAST15_FEATURE_COLUMNS
    )
    if len(feature_columns) != 307:
        raise AssertionError("Unexpected EXP067 feature count.")
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Duplicate features entered EXP067.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    raw_target = model_data.loc[train_mask, "target"].to_numpy(dtype=np.float64)
    target_mean = float(raw_target.mean())
    centered_target = raw_target - target_mean
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    parameters = {
        "objective": "regression",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": 2,
        "verbosity": -1,
        "force_col_wise": True,
        "deterministic": True,
    }
    model = LGBMRegressor(**parameters)
    print("training EXP-TREE-067 LightGBM on 307 EXP053R features", flush=True)
    start_time = time.perf_counter()
    model.fit(model_data.loc[train_mask, feature_columns], centered_target)
    training_seconds = time.perf_counter() - start_time
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id="EXP-TREE-067",
        description=(
            "LightGBM fixed-800 comparison on the complete 307-feature EXP053R "
            "schema with centered target and two CPU threads"
        ),
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-053R",
        save_model=False,
    )
    finalize_metadata(
        project_dir=project_dir,
        experiment_id="EXP-TREE-067",
        metadata=metadata,
        baseline_predictions=baseline_predictions,
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )

    model_path = project_dir / "outputs" / "models" / "exp-tree-067.joblib"
    joblib.dump(model, model_path)
    metadata["model_family"] = "LightGBM"
    metadata["lightgbm_version"] = lightgbm.__version__
    metadata["training_target_mean"] = target_mean
    metadata["training_device"] = "cpu-2-threads"
    metadata["model_path"] = str(model_path)
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-067"
        / "config.json"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
