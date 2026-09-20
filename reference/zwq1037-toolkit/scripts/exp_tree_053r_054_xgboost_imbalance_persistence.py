"""Run an exact EXP053 GPU control and add three imbalance path features."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_imbalance_persistence_features import (
    FEATURE_COLUMNS as PERSISTENCE_FEATURE_COLUMNS,
)
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


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    public_features, dropped_public_features = selected_public_features(project_dir)

    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    processed_dir = project_dir / "data" / "processed"
    gap = pd.read_feather(processed_dir / "train_transaction_event_gap_features.feather")
    last15 = pd.read_feather(processed_dir / "train_market_last15_baseline_features.feather")
    persistence = pd.read_feather(
        processed_dir / "train_market_imbalance_persistence_features.feather"
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
        .merge(persistence, on="sample_id", how="left", validate="one_to_one")
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
    del gap, last15, persistence, public, labels
    gc.collect()

    exp051_features = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
    )
    control_features = exp051_features + LAST15_FEATURE_COLUMNS
    candidate_features = control_features + PERSISTENCE_FEATURE_COLUMNS
    if len(control_features) != 307 or len(candidate_features) != 310:
        raise AssertionError("Unexpected EXP053R/EXP054 feature count.")
    if len(candidate_features) != len(set(candidate_features)):
        raise AssertionError("Duplicate features entered EXP054.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    previous_exp053 = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053_valid.feather"
    )
    assert_prediction_alignment(previous_exp053, validation_rows)

    print("training EXP-TREE-053R exact-ratio GPU control", flush=True)
    control_predictions = run_one(
        project_dir=project_dir,
        experiment_id="EXP-TREE-053R",
        description=(
            "EXP053 reproduced with the final-15/full RV ratio using the complete "
            "raw-sequence m_rv denominator"
        ),
        model_data=model_data,
        feature_columns=control_features,
        train_mask=train_mask,
        valid_mask=valid_mask,
        centered_target=centered_target,
        validation_rows=validation_rows,
        baseline_predictions=previous_exp053,
        baseline_id="EXP-TREE-053",
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )
    print("training EXP-TREE-054 imbalance persistence candidate", flush=True)
    run_one(
        project_dir=project_dir,
        experiment_id="EXP-TREE-054",
        description=(
            "EXP053R plus three time-weighted final-60-second order-imbalance "
            "occupancy, flip-rate, and signed terminal-run features"
        ),
        model_data=model_data,
        feature_columns=candidate_features,
        train_mask=train_mask,
        valid_mask=valid_mask,
        centered_target=centered_target,
        validation_rows=validation_rows,
        baseline_predictions=control_predictions,
        baseline_id="EXP-TREE-053R",
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )


if __name__ == "__main__":
    main()
