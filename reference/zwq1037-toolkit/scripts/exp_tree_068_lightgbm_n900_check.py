"""Evaluate the first 900 trees of the trained 1,000-tree EXP068 model."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd

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
        raise AssertionError("Unexpected EXP068-N900 feature count.")

    valid_mask = model_data["month"] >= 60
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    validation_features = model_data.loc[valid_mask, feature_columns]
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    base_run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-068"
    )
    base_config = json.loads(
        (base_run_dir / "config.json").read_text(encoding="utf-8")
    )
    if base_config["feature_columns"] != feature_columns:
        raise AssertionError("EXP068 model feature order differs from current data.")
    model_path = project_dir / "outputs" / "models" / "exp-tree-068.joblib"
    model = joblib.load(model_path)

    prediction_start = time.perf_counter()
    predictions = model.predict(validation_features, num_iteration=900)
    prediction_seconds = time.perf_counter() - prediction_start
    parameters = dict(base_config["parameters"])
    parameters["prediction_iteration"] = 900
    experiment_id = "EXP-TREE-068-N900"
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=experiment_id,
        description="Prefix evaluation of EXP068 using its first 900 of 1000 trees",
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=base_config["training_seconds"],
        baseline_id="EXP-TREE-053R",
        save_model=False,
    )
    finalize_metadata(
        project_dir=project_dir,
        experiment_id=experiment_id,
        metadata=metadata,
        baseline_predictions=baseline_predictions,
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )
    metadata["model_family"] = "LightGBM"
    metadata["lightgbm_version"] = lightgbm.__version__
    metadata["training_device"] = "cpu-2-threads"
    metadata["model_path"] = str(model_path)
    metadata["shared_training_run"] = "EXP-TREE-068"
    metadata["prediction_iteration"] = 900
    metadata["prediction_seconds"] = prediction_seconds
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / experiment_id
        / "config.json"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{experiment_id}: overall={metadata['overall_cosine']:.9f}, "
        f"no66={metadata['cosine_62_70_without_66']:.9f}, "
        f"recent={metadata['cosine_67_70']:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
