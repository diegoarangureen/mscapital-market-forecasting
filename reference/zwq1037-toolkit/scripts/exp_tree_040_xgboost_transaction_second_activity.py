"""Test transaction second-activity features on EXP-TREE-034."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_transaction_second_activity_features import FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import assert_prediction_alignment
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import (
    add_event_features,
    evaluate_and_save,
    fit_model,
    model_parameters,
)
from train_full_xgboost_submissions import load_features


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    activity = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / "train_transaction_second_activity_features.feather"
    )
    labels = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    model_data = (
        model_data.merge(activity, on="sample_id", how="left", validate="one_to_one")
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
    )
    del activity, labels
    gc.collect()

    baseline_features = base_features + TRANSACTION_FEATURE_COLUMNS + ORDER_MULTI_FEATURE_COLUMNS
    feature_columns = baseline_features + FEATURE_COLUMNS
    if len(baseline_features) != 155 or len(feature_columns) != 159:
        raise AssertionError("Unexpected EXP-TREE-040 feature schema.")
    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_rows = model_data.loc[valid_mask, ["sample_id", "month", "target"]].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-034_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    model, training_seconds = fit_model(
        model_data, feature_columns, train_mask, centered_target, parameters
    )
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    evaluate_and_save(
        project_dir=project_dir,
        experiment_id="EXP-TREE-040",
        description="add transaction active-second ratios and maximum inactive-run ratio",
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-034",
    )


if __name__ == "__main__":
    main()
