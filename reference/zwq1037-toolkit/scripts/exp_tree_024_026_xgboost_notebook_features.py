"""Controlled XGBoost ablations for public-notebook-inspired feature groups."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import xgboost

from build_event_flow_features import ORDER_FEATURE_COLUMNS, TRANSACTION_FEATURE_COLUMNS
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
    compare_months,
    cosine_similarity_score,
    fit_one,
    save_model_safely,
)


TRANSACTION_MULTI_FEATURE_COLUMNS = [
    "trade_volume_imbalance_10",
    "trade_volume_imbalance_30",
    "trade_count_imbalance_10",
    "trade_count_imbalance_30",
    "trade_signed_amount_ratio_10",
    "trade_signed_amount_ratio_30",
    "trade_signed_amount_ratio_60",
    "trade_vwap_10_vs_60",
    "trade_vwap_30_vs_60",
    "trade_volume_share_10_of_60",
    "trade_volume_share_30_of_60",
    "trade_count_share_10_of_60",
    "trade_count_share_30_of_60",
    "trade_pressure_delta_10_vs_10_30",
    "trade_pressure_delta_30_vs_30_60",
]
ORDER_MULTI_FEATURE_COLUMNS = [
    "new_order_volume_imbalance_10",
    "new_order_volume_imbalance_30",
    "cancel_order_pressure_10",
    "cancel_order_pressure_30",
    "net_order_pressure_10",
    "net_order_pressure_30",
    "total_cancel_ratio_10",
    "total_cancel_ratio_30",
    "order_volume_share_10_of_60",
    "order_volume_share_30_of_60",
    "order_count_share_10_of_60",
    "order_count_share_30_of_60",
    "net_order_pressure_delta_10_vs_10_30",
    "net_order_pressure_delta_30_vs_30_60",
    "cancel_ratio_delta_10_vs_60",
]
DECAY_SIGNALS = [
    "book_imbalance_1",
    "total_book_imbalance",
    "microprice_displacement",
    "relative_spread",
    "ofi_1",
    "multilevel_ofi",
    "log_total_depth",
]
MARKET_DECAY_FEATURE_COLUMNS = [
    name
    for signal in DECAY_SIGNALS
    for name in (
        f"{signal}_ewm_30",
        f"{signal}_ewm_120",
        f"{signal}_ewm_30_minus_120",
    )
]

VARIANTS = {
    "EXP-TREE-024": {
        "name": "transaction_multiwindow",
        "description": "add 15 nested 10/30/60-second transaction features to EXP-TREE-020",
        "tables": ["transaction_multi"],
        "added_features": TRANSACTION_MULTI_FEATURE_COLUMNS,
    },
    "EXP-TREE-025": {
        "name": "order_multiwindow",
        "description": "add the existing order group plus 15 nested 10/30/60-second order features to EXP-TREE-020",
        "tables": ["order", "order_multi"],
        "added_features": ORDER_FEATURE_COLUMNS + ORDER_MULTI_FEATURE_COLUMNS,
    },
    "EXP-TREE-026": {
        "name": "market_exponential_decay",
        "description": "add 21 tau-30/tau-120 exponentially decayed market signals to EXP-TREE-020",
        "tables": ["market_decay"],
        "added_features": MARKET_DECAY_FEATURE_COLUMNS,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", choices=sorted(VARIANTS))
    return parser.parse_args()


def subset_score(predictions: pd.DataFrame, query: str) -> dict:
    subset = predictions.query(query)
    return {
        "query": query,
        "row_count": int(len(subset)),
        "cosine": cosine_similarity_score(subset["target"], subset["prediction"]),
    }


def main() -> None:
    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"

    v1_data = pd.read_feather(processed_dir / "train_market_features.feather")
    last60_data = pd.read_feather(processed_dir / "train_market_last60_features_complete.feather")
    level2_data = pd.read_feather(processed_dir / "train_market_level2_features.feather")
    micro_data = pd.read_feather(
        processed_dir / "train_market_microstructure_features.feather",
        columns=["sample_id", *BOOK_FEATURE_COLUMNS],
    )
    transaction_data = pd.read_feather(processed_dir / "train_transaction_flow_features.feather")
    label_data = pd.read_feather(project_dir / "data" / "raw" / "label.feather")

    base_features = (
        v1_data.columns[1:].tolist()
        + last60_data.columns[1:].tolist()
        + level2_data.columns[1:].tolist()
        + BOOK_FEATURE_COLUMNS
    )
    if len(base_features) != 132 or len(set(base_features)) != 132:
        raise AssertionError("Expected the exact 132-feature EXP-TREE-016 baseline.")
    common_features = base_features + TRANSACTION_FEATURE_COLUMNS
    if len(common_features) != 140 or len(set(common_features)) != 140:
        raise AssertionError("Expected the exact 140-feature EXP-TREE-020 baseline.")

    common_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(level2_data, on="sample_id", how="left", validate="one_to_one")
        .merge(micro_data, on="sample_id", how="left", validate="one_to_one")
        .merge(transaction_data, on="sample_id", how="left", validate="one_to_one")
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    common_data["last60_row_count"] = common_data["last60_row_count"].fillna(0)
    del v1_data, last60_data, level2_data, micro_data, transaction_data, label_data
    gc.collect()

    table_data = {
        "transaction_multi": pd.read_feather(processed_dir / "train_transaction_multiwindow_features.feather"),
        "order": pd.read_feather(processed_dir / "train_order_flow_features.feather"),
        "order_multi": pd.read_feather(processed_dir / "train_order_multiwindow_features.feather"),
        "market_decay": pd.read_feather(processed_dir / "train_market_decay_features.feather"),
    }
    for name, frame in table_data.items():
        if len(frame) != len(common_data) or frame["sample_id"].nunique() != len(frame):
            raise AssertionError(f"Malformed feature table: {name}")

    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "xgboost_microstructure_transaction_flow_valid.feather"
    )
    baseline_score_62_70 = subset_score(baseline_predictions, "month >= 62")
    baseline_score_62_70_without_66 = subset_score(
        baseline_predictions, "month >= 62 and month != 66"
    )
    order_baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "xgboost_microstructure_transaction_and_order_flow_valid.feather"
    )
    baseline_overall = cosine_similarity_score(
        baseline_predictions["target"], baseline_predictions["prediction"]
    )

    selected = VARIANTS
    if arguments.experiment_id:
        selected = {arguments.experiment_id: VARIANTS[arguments.experiment_id]}

    for experiment_id, spec in selected.items():
        model_data = common_data
        for table_name in spec["tables"]:
            model_data = model_data.merge(
                table_data[table_name], on="sample_id", how="left", validate="one_to_one"
            )
        feature_columns = common_features + spec["added_features"]
        if len(feature_columns) != len(set(feature_columns)):
            raise AssertionError(f"Duplicate features in {experiment_id}.")
        if {"sample_id", "month", "target"}.intersection(feature_columns):
            raise AssertionError("Identity, time, or target entered X.")

        train_mask = model_data["month"] <= 59
        valid_mask = model_data["month"] >= 60
        model, predictions, monthly_scores, metadata = fit_one(
            model_data, feature_columns, train_mask, valid_mask, target_centered=True
        )
        assert_prediction_alignment(baseline_predictions, predictions)
        comparison = compare_months(baseline_predictions, monthly_scores)
        score_62_70 = subset_score(predictions, "month >= 62")
        score_62_70_without_66 = subset_score(predictions, "month >= 62 and month != 66")

        run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = project_dir / "outputs" / "predictions" / f"xgboost_{spec['name']}_valid.feather"
        model_path = project_dir / "outputs" / "models" / f"xgboost_{spec['name']}.json"
        predictions.to_feather(prediction_path)
        monthly_scores.to_csv(run_dir / "monthly_cosine.csv", index=False)
        comparison.to_csv(run_dir / "monthly_comparison.csv", index=False)
        save_model_safely(model, model_path)

        config = {
            "experiment_id": experiment_id,
            "description": spec["description"],
            "baseline": "EXP-TREE-020: 132 market features plus 8 transaction-flow features",
            "feature_count": len(feature_columns),
            "added_feature_columns": spec["added_features"],
            "target_centered": True,
            "train_months": "0-59",
            "validation_months": "60-70",
            "official_metric": "uncentered whole-vector cosine",
            "xgboost_version": xgboost.__version__,
            "formal_metadata": metadata,
            "robustness_62_70": score_62_70,
            "robustness_62_70_without_66": score_62_70_without_66,
            "baseline_62_70": baseline_score_62_70,
            "baseline_62_70_without_66": baseline_score_62_70_without_66,
            "change_from_exp020": metadata["overall_cosine"] - baseline_overall,
            "change_62_70_from_exp020": score_62_70["cosine"] - baseline_score_62_70["cosine"],
            "change_62_70_without_66_from_exp020": (
                score_62_70_without_66["cosine"] - baseline_score_62_70_without_66["cosine"]
            ),
            "improved_month_count_from_exp020": int((comparison["change"] > 0).sum()),
            "declined_month_count_from_exp020": int((comparison["change"] < 0).sum()),
            "prediction_path": str(prediction_path),
            "model_path": str(model_path),
        }
        if experiment_id == "EXP-TREE-025":
            assert_prediction_alignment(order_baseline_predictions, predictions)
            config["change_from_exp022"] = metadata["overall_cosine"] - cosine_similarity_score(
                order_baseline_predictions["target"], order_baseline_predictions["prediction"]
            )
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{experiment_id}: overall={metadata['overall_cosine']:.10f}, "
            f"62-70={score_62_70['cosine']:.10f}, "
            f"62-70_no66={score_62_70_without_66['cosine']:.10f}, "
            f"monthly_std={metadata['monthly_stability']['monthly_population_std']:.10f}, "
            f"improved_months={config['improved_month_count_from_exp020']}/11",
            flush=True,
        )
        del model, predictions, monthly_scores, comparison, model_data, train_mask, valid_mask
        gc.collect()


if __name__ == "__main__":
    main()
