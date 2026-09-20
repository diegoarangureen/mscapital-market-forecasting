"""Test mild linear recency weights on robust EXP-TREE-037."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
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


def build_recency_weights(months: pd.Series) -> np.ndarray:
    """Map training months linearly from weight 1.0 to 1.5."""

    values = months.to_numpy(dtype=np.float64)
    span = float(values.max() - values.min())
    if span <= 0:
        return np.ones(len(values), dtype=np.float64)
    return 1.0 + 0.5 * (values - values.min()) / span


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
        raise AssertionError("Unexpected EXP-TREE-046 feature schema.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    training_target = model_data.loc[train_mask, "target"].to_numpy(dtype=np.float64)
    sample_weight = build_recency_weights(model_data.loc[train_mask, "month"])
    weighted_target_mean = float(np.average(training_target, weights=sample_weight))
    centered_target = pd.Series(
        training_target - weighted_target_mean,
        index=model_data.index[train_mask],
    )
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-037_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    model, training_seconds = fit_model(
        model_data,
        feature_columns,
        train_mask,
        centered_target,
        parameters,
        sample_weight=sample_weight,
    )
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id="EXP-TREE-046",
        description="apply mild linear month weights from 1.0 to 1.5 to robust EXP037",
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-037",
    )
    metadata["sample_weight_min"] = float(sample_weight.min())
    metadata["sample_weight_max"] = float(sample_weight.max())
    metadata["weighted_target_mean"] = weighted_target_mean
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-046"
        / "config.json"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
