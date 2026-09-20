"""Add directional agreement strengths across recent book, OFI, and trade flow."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import add_event_features
from exp_tree_052_053_xgboost_gpu_last15 import run_one
from train_full_xgboost_exp051_submission import (
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


ADDED_FEATURES = [
    "x_book_trade_aligned_pressure_20",
    "x_book_ofi_aligned_pressure_20",
    "x_ofi_trade_aligned_pressure_20",
]


def aligned_pressure(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_values = left.to_numpy(dtype=np.float64)
    right_values = right.to_numpy(dtype=np.float64)
    product = left_values * right_values
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    aligned = finite & (product > 0.0)
    output = np.full(len(left_values), np.nan, dtype=np.float64)
    output[finite] = 0.0
    output[aligned] = (
        np.sign(left_values[aligned] + right_values[aligned])
        * np.minimum(
            np.abs(left_values[aligned]),
            np.abs(right_values[aligned]),
        )
    )
    return output


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

    book_imbalance = model_data["book_imbalance_1_20_mean_robust"]
    ofi = model_data["ofi_1_20_mean"]
    trade_imbalance = model_data["trade_volume_imbalance_20"]
    model_data[ADDED_FEATURES[0]] = aligned_pressure(
        book_imbalance,
        trade_imbalance,
    )
    model_data[ADDED_FEATURES[1]] = aligned_pressure(book_imbalance, ofi)
    model_data[ADDED_FEATURES[2]] = aligned_pressure(ofi, trade_imbalance)
    del gap, last15, public, labels
    gc.collect()

    feature_columns = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
        + LAST15_FEATURE_COLUMNS
        + ADDED_FEATURES
    )
    if len(feature_columns) != 310:
        raise AssertionError("Unexpected EXP066 feature count.")
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Duplicate features entered EXP066.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    print("training EXP-TREE-066 recent flow-alignment features", flush=True)
    run_one(
        project_dir=project_dir,
        experiment_id="EXP-TREE-066",
        description=(
            "EXP053R plus three directional aligned-pressure strengths between "
            "final-20-second book imbalance, OFI, and trade imbalance"
        ),
        model_data=model_data,
        feature_columns=feature_columns,
        train_mask=train_mask,
        valid_mask=valid_mask,
        centered_target=centered_target,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        baseline_id="EXP-TREE-053R",
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )


if __name__ == "__main__":
    main()
