"""Test three pruned market-decay signals on robust EXP-TREE-037."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as TRANSACTION_GAP_FEATURES
from exp_tree_017_019_xgboost_target_and_monthly_transforms import assert_prediction_alignment
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import (
    add_event_features,
    evaluate_and_save,
    fit_model,
    model_parameters,
)
from train_full_xgboost_submissions import load_features


PRUNED_MARKET_DECAY_FEATURES = [
    "microprice_displacement_ewm_30_minus_120",
    "ofi_1_ewm_30",
    "ofi_1_ewm_30_minus_120",
]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    transaction_gaps = pd.read_feather(
        processed_dir / "train_transaction_event_gap_features.feather"
    )
    market_decay = pd.read_feather(
        processed_dir / "train_market_decay_features.feather",
        columns=["sample_id", *PRUNED_MARKET_DECAY_FEATURES],
    )
    labels = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    model_data = (
        model_data.merge(
            transaction_gaps, on="sample_id", how="left", validate="one_to_one"
        )
        .merge(market_decay, on="sample_id", how="left", validate="one_to_one")
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
    )
    del transaction_gaps, market_decay, labels
    gc.collect()

    exp034_features = base_features + TRANSACTION_FEATURE_COLUMNS + ORDER_MULTI_FEATURE_COLUMNS
    exp037_features = exp034_features + TRANSACTION_GAP_FEATURES
    feature_columns = exp037_features + PRUNED_MARKET_DECAY_FEATURES
    if len(exp037_features) != 159 or len(feature_columns) != 162:
        raise AssertionError("Unexpected EXP-TREE-042 feature schema.")
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Duplicate feature names in EXP-TREE-042.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-037_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    model, training_seconds = fit_model(
        model_data, feature_columns, train_mask, centered_target, parameters
    )
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    evaluate_and_save(
        project_dir=project_dir,
        experiment_id="EXP-TREE-042",
        description=(
            "add pruned microprice displacement short-long decay, 30-second OFI, "
            "and OFI short-long decay to robust EXP037"
        ),
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-037",
    )


if __name__ == "__main__":
    main()
