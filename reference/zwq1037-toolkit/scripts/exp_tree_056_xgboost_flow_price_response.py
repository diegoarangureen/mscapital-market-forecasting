"""Test two cheap final-60-second flow-versus-price-response interactions."""

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


RESPONSE_CLIP = 5.0
RESPONSE_FEATURE_COLUMNS = [
    "flow_price_alignment_60",
    "flow_without_price_move_60",
]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    public_features, dropped_public_features = selected_public_features(project_dir)
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    processed_dir = project_dir / "data" / "processed"
    gap = pd.read_feather(processed_dir / "train_transaction_event_gap_features.feather")
    last15 = pd.read_feather(processed_dir / "train_market_last15_baseline_features.feather")
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

    signed_flow = model_data["trade_volume_imbalance_60"].fillna(0.0).to_numpy(
        dtype=np.float64
    )
    mid_change = model_data["last60_mid_price_1_change"].to_numpy(dtype=np.float64)
    spread = model_data["last60_spread_1_mean"].to_numpy(dtype=np.float64)
    response_in_spreads = np.divide(
        mid_change,
        spread,
        out=np.zeros(len(model_data), dtype=np.float64),
        where=np.isfinite(spread) & (np.abs(spread) > 1e-8),
    )
    response_in_spreads = np.nan_to_num(
        response_in_spreads, nan=0.0, posinf=0.0, neginf=0.0
    )
    model_data["flow_price_alignment_60"] = (
        signed_flow * np.clip(response_in_spreads, -RESPONSE_CLIP, RESPONSE_CLIP)
    ).astype(np.float32)
    model_data["flow_without_price_move_60"] = (
        signed_flow / (1.0 + np.abs(response_in_spreads))
    ).astype(np.float32)
    del gap, last15, public, labels
    gc.collect()

    control_features = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
        + LAST15_FEATURE_COLUMNS
    )
    candidate_features = control_features + RESPONSE_FEATURE_COLUMNS
    if len(control_features) != 307 or len(candidate_features) != 309:
        raise AssertionError("Unexpected EXP056 feature count.")
    if len(candidate_features) != len(set(candidate_features)):
        raise AssertionError("Duplicate features entered EXP056.")

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

    print("training EXP-TREE-056 flow-price-response candidate", flush=True)
    run_one(
        project_dir=project_dir,
        experiment_id="EXP-TREE-056",
        description=(
            "EXP053R plus final-60-second signed-flow alignment with spread-scaled "
            "mid-price response and signed flow without price movement"
        ),
        model_data=model_data,
        feature_columns=candidate_features,
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
