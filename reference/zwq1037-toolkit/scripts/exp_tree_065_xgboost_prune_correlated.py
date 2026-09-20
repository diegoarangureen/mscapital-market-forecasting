"""Remove 14 training-only near-duplicate features from EXP053R."""

from __future__ import annotations

import gc
import json
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


EXPERIMENT_ID = "EXP-TREE-065"


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

    full_features = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
        + LAST15_FEATURE_COLUMNS
    )
    if len(full_features) != 307 or len(full_features) != len(set(full_features)):
        raise AssertionError("Unexpected EXP053R feature schema.")

    selection_path = (
        project_dir
        / "data"
        / "interim"
        / "exp053r_correlated_feature_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_source"] != "training months 0-59 only":
        raise AssertionError("Correlation selection did not use the training-only audit.")
    if selection["correlation_threshold"] != 0.995:
        raise AssertionError("Unexpected correlation threshold.")
    removed_features = [
        pair["removed_feature"] for pair in selection["removed_pairs"]
    ]
    if len(removed_features) != 14 or len(removed_features) != len(set(removed_features)):
        raise AssertionError("Unexpected correlated-feature removal count.")
    removed_set = set(removed_features)
    feature_columns = [
        feature for feature in full_features if feature not in removed_set
    ]
    if len(feature_columns) != 293:
        raise AssertionError("Unexpected EXP065 feature count.")

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

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

    print(
        f"training {EXPERIMENT_ID}: kept={len(feature_columns)}, "
        f"correlated_dropped={len(removed_features)}",
        flush=True,
    )
    run_one(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "EXP053R after removing 14 near-duplicate features selected at absolute "
            "Pearson correlation >= 0.995 on equal samples from training months 0-59"
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
