"""Test deployable per-row ranks across EXP-TREE-037's leakage-safe top features."""

from __future__ import annotations

import gc
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


# 这些列来自只使用 0～59 月训练的 EXP037 模型，不读取验证标签。
# These columns come from EXP037 trained only on months 0-59, without validation labels.
TOP_FEATURES = [
    "trade_volume_imbalance_20",
    "last60_ask_price_1_end",
    "ofi_1_20_mean",
    "trade_count_imbalance_60",
    "new_order_volume_imbalance_10",
    "net_order_pressure_30",
    "ofi_1_last",
    "microprice_displacement_last",
    "bid_price_1_last",
    "ask_price_1_last",
    "trade_volume_imbalance_60",
    "net_order_pressure_10",
    "last_trade_seconds_before_predict",
    "microprice_displacement_20_mean",
    "spread_1_mean",
    "cancel_order_pressure_10",
    "last60_spread_2_mean",
    "last60_bid_price_2_end",
    "bid_price_2_last",
    "bid_price_2_std",
    "ask_volume_1_mean",
    "book_imbalance_1_std_robust",
    "log1p_total_trade_volume_60",
    "last60_bid_volume_2_mean",
    "last60_bid_price_2_start",
    "order_volume_share_30_of_60",
    "cancel_order_pressure_30",
    "ask_volume_1_sum",
    "order_count_share_10_of_60",
    "ask_volume_2_sum",
]
CROSS_RANK_FEATURES = [f"cross_rank_top_{index:02d}" for index in range(len(TOP_FEATURES))]


def add_cross_feature_ranks(data: pd.DataFrame, batch_size: int = 50_000) -> pd.DataFrame:
    """Rank the selected feature values within each row on a fixed [-0.5, 0.5] grid."""

    output = np.empty((len(data), len(TOP_FEATURES)), dtype=np.float32)
    rank_grid = np.linspace(-0.5, 0.5, len(TOP_FEATURES), dtype=np.float32)
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        values = data.iloc[start:end][TOP_FEATURES].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(values, axis=1, kind="stable")
        ranks = np.empty(order.shape, dtype=np.float32)
        row_indices = np.arange(end - start)[:, None]
        ranks[row_indices, order] = rank_grid
        output[start:end] = ranks
    rank_frame = pd.DataFrame(output, columns=CROSS_RANK_FEATURES, index=data.index)
    return pd.concat([data, rank_frame], axis=1)


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
    exp037_features = exp034_features + TRANSACTION_GAP_FEATURES
    if len(exp037_features) != 159 or not set(TOP_FEATURES).issubset(exp037_features):
        raise AssertionError("Unexpected EXP-TREE-043 base feature schema.")
    model_data = add_cross_feature_ranks(model_data)
    feature_columns = exp037_features + CROSS_RANK_FEATURES
    if len(feature_columns) != 189 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Unexpected EXP-TREE-043 final feature schema.")

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
        experiment_id="EXP-TREE-043",
        description=(
            "add 30 deployable per-row cross-feature ranks derived from the leakage-safe "
            "EXP037 training-model importance"
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
