"""Controlled XGBoost ablations for transaction and order event flows."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import xgboost

from build_event_flow_features import ORDER_FEATURE_COLUMNS, TRANSACTION_FEATURE_COLUMNS
from build_event_market_cross_features import CROSS_FEATURE_COLUMNS
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
    compare_months,
    fit_one,
    save_model_safely,
)


VARIANTS = {
    "EXP-TREE-020": {
        "name": "microstructure_transaction_flow",
        "description": "add 8 active-buy/active-sell transaction-flow features only",
        "event_features": TRANSACTION_FEATURE_COLUMNS,
    },
    "EXP-TREE-021": {
        "name": "microstructure_order_flow",
        "description": "add 8 new/cancel buy/sell order-flow features only",
        "event_features": ORDER_FEATURE_COLUMNS,
    },
    "EXP-TREE-022": {
        "name": "microstructure_transaction_and_order_flow",
        "description": "combine the separately tested transaction and order-flow groups",
        "event_features": TRANSACTION_FEATURE_COLUMNS + ORDER_FEATURE_COLUMNS,
    },
    "EXP-TREE-023": {
        "name": "microstructure_transaction_flow_depth_cross",
        "description": "add 5 trade-pressure versus visible-depth interactions to EXP-TREE-020",
        "event_features": TRANSACTION_FEATURE_COLUMNS + CROSS_FEATURE_COLUMNS,
    },
}


def parse_arguments():
    """Require one explicit experiment so combined features are never run accidentally."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True, choices=sorted(VARIANTS))
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    experiment_id = arguments.experiment_id
    spec = VARIANTS[experiment_id]
    project_dir = Path(__file__).resolve().parents[1]

    v1_data = pd.read_feather(project_dir / "data" / "processed" / "train_market_features.feather")
    last60_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_last60_features_complete.feather"
    )
    level2_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_level2_features.feather"
    )
    micro_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_microstructure_features.feather",
        columns=["sample_id", *BOOK_FEATURE_COLUMNS],
    )
    transaction_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_transaction_flow_features.feather"
    )
    order_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_order_flow_features.feather"
    )
    cross_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_event_market_cross_features.feather"
    )
    label_data = pd.read_feather(project_dir / "data" / "raw" / "label.feather")

    base_features = (
        v1_data.columns[1:].tolist()
        + last60_data.columns[1:].tolist()
        + level2_data.columns[1:].tolist()
        + BOOK_FEATURE_COLUMNS
    )
    if len(base_features) != 132 or len(set(base_features)) != 132:
        raise AssertionError("Expected the exact 132-feature EXP-TREE-016 baseline.")
    event_features = spec["event_features"]
    feature_columns = base_features + event_features
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Duplicate event feature names.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Identity, time, or target entered X.")

    model_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(level2_data, on="sample_id", how="left", validate="one_to_one")
        .merge(micro_data, on="sample_id", how="left", validate="one_to_one")
        .merge(transaction_data, on="sample_id", how="left", validate="one_to_one")
        .merge(order_data, on="sample_id", how="left", validate="one_to_one")
        .merge(cross_data, on="sample_id", how="left", validate="one_to_one")
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)
    del v1_data, last60_data, level2_data, micro_data, transaction_data, order_data, cross_data, label_data
    gc.collect()

    baseline_id = "EXP-TREE-016"
    baseline_internal = pd.read_feather(
        project_dir / "data" / "interim" / "tree_experiments" / baseline_id / "internal_predictions.feather"
    )
    baseline_formal = pd.read_feather(
        project_dir / "outputs" / "predictions" / "xgboost_book_microstructure_valid.feather"
    )
    baseline_config = json.loads(
        (project_dir / "data" / "interim" / "tree_experiments" / baseline_id / "config.json").read_text(
            encoding="utf-8"
        )
    )
    internal_train = model_data["month"] <= 49
    internal_valid = model_data["month"].between(50, 59)
    formal_train = model_data["month"] <= 59
    formal_valid = model_data["month"] >= 60
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    internal_model, internal_predictions, internal_monthly, internal_metadata = fit_one(
        model_data,
        feature_columns,
        internal_train,
        internal_valid,
        target_centered=True,
    )
    del internal_model
    assert_prediction_alignment(baseline_internal, internal_predictions)
    internal_comparison = compare_months(baseline_internal, internal_monthly)
    internal_predictions.to_feather(run_dir / "internal_predictions.feather")
    internal_monthly.to_csv(run_dir / "internal_monthly_cosine.csv", index=False)
    internal_comparison.to_csv(run_dir / "internal_monthly_comparison.csv", index=False)
    print(
        f"{experiment_id} internal: cosine={internal_metadata['overall_cosine']:.10f}, "
        f"change={internal_metadata['overall_cosine'] - baseline_config['internal_metadata']['overall_cosine']:+.10f}",
        flush=True,
    )
    del internal_predictions, internal_monthly, internal_comparison
    gc.collect()

    formal_model, formal_predictions, formal_monthly, formal_metadata = fit_one(
        model_data,
        feature_columns,
        formal_train,
        formal_valid,
        target_centered=True,
    )
    assert_prediction_alignment(baseline_formal, formal_predictions)
    formal_comparison = compare_months(baseline_formal, formal_monthly)
    prediction_path = project_dir / "outputs" / "predictions" / f"xgboost_{spec['name']}_valid.feather"
    model_path = project_dir / "outputs" / "models" / f"xgboost_{spec['name']}.json"
    formal_predictions.to_feather(prediction_path)
    formal_monthly.to_csv(run_dir / "monthly_cosine.csv", index=False)
    formal_comparison.to_csv(run_dir / "monthly_comparison.csv", index=False)
    save_model_safely(formal_model, model_path)
    config = {
        "experiment_id": experiment_id,
        "description": spec["description"],
        "baseline": "EXP-TREE-016 XGBoost with 132 deployable market features",
        "main_change": spec["description"],
        "feature_count": len(feature_columns),
        "added_feature_columns": event_features,
        "target_centered": True,
        "train_months": "0-59",
        "validation_months": "60-70",
        "internal_split": "train 0-49, validate 50-59",
        "official_metric": "uncentered whole-vector cosine",
        "event_encoding": {
            "transaction_side": "0=aggressive buy, 1=aggressive sell",
            "order_side": "0=buy, 1=sell",
            "order_action": "0=new, 1=cancel",
        },
        "xgboost_version": xgboost.__version__,
        "internal_metadata": internal_metadata,
        "formal_metadata": formal_metadata,
        "internal_baseline_cosine": baseline_config["internal_metadata"]["overall_cosine"],
        "formal_baseline_cosine": baseline_config["formal_metadata"]["overall_cosine"],
        "internal_change_from_baseline": internal_metadata["overall_cosine"] - baseline_config["internal_metadata"]["overall_cosine"],
        "formal_change_from_baseline": formal_metadata["overall_cosine"] - baseline_config["formal_metadata"]["overall_cosine"],
        "formal_improved_month_count": int((formal_comparison["change"] > 0).sum()),
        "formal_declined_month_count": int((formal_comparison["change"] < 0).sum()),
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{experiment_id} formal: cosine={formal_metadata['overall_cosine']:.10f}, "
        f"change={formal_metadata['overall_cosine'] - baseline_config['formal_metadata']['overall_cosine']:+.10f}, "
        f"monthly_std={formal_metadata['monthly_stability']['monthly_population_std']:.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
