"""Find near-duplicate EXP053R features using training-month samples only."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import add_event_features
from train_full_xgboost_exp051_submission import (
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


CORRELATION_THRESHOLD = 0.995
SAMPLES_PER_MONTH = 1000


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    public_features, _ = selected_public_features(project_dir)
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    gap = pd.read_feather(processed_dir / "train_transaction_event_gap_features.feather")
    last15 = pd.read_feather(
        processed_dir / "train_market_last15_baseline_features.feather"
    )
    public = load_public_table(project_dir, "train", public_features).drop(
        columns="target"
    )
    months = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month"],
    )
    model_data = (
        model_data.merge(gap, on="sample_id", how="left", validate="one_to_one")
        .merge(public, on="sample_id", how="left", validate="one_to_one")
        .merge(last15, on="sample_id", how="left", validate="one_to_one")
        .merge(months, on="sample_id", how="inner", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    model_data["x_rv_15_over_full"] = model_data["m_rv_15"] / (
        model_data["m_rv"] + 1e-8
    )
    del gap, last15, public, months
    gc.collect()

    feature_columns = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
        + LAST15_FEATURE_COLUMNS
    )
    if len(feature_columns) != 307 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Unexpected EXP053R feature schema.")

    # 每个训练月等量抽样，避免相关性结构被样本较多的月份主导。
    # Sample each training month equally so large months do not dominate correlation.
    random_generator = np.random.default_rng(42)
    sampled_indices: list[np.ndarray] = []
    for month in range(60):
        month_indices = np.flatnonzero(model_data["month"].to_numpy() == month)
        sample_size = min(SAMPLES_PER_MONTH, len(month_indices))
        sampled_indices.append(
            random_generator.choice(month_indices, size=sample_size, replace=False)
        )
    selected_indices = np.concatenate(sampled_indices)
    feature_sample = model_data.loc[selected_indices, feature_columns].astype(
        np.float32
    )
    correlation = feature_sample.corr(method="pearson", min_periods=200).abs()
    del feature_sample, model_data
    gc.collect()

    baseline_model = xgb.XGBRegressor()
    baseline_model.load_model(
        project_dir / "outputs" / "models" / "exp-tree-053r.json"
    )
    total_gain = baseline_model.get_booster().get_score(
        importance_type="total_gain"
    )
    ranked_features = sorted(
        feature_columns,
        key=lambda feature: (-total_gain.get(feature, 0.0), feature),
    )
    kept_features: list[str] = []
    removed: list[dict[str, float | str]] = []
    for feature in ranked_features:
        if not kept_features:
            kept_features.append(feature)
            continue
        correlations_to_kept = correlation.loc[feature, kept_features]
        highest_correlation = float(correlations_to_kept.max(skipna=True))
        if highest_correlation >= CORRELATION_THRESHOLD:
            keeper = str(correlations_to_kept.idxmax(skipna=True))
            removed.append(
                {
                    "removed_feature": feature,
                    "kept_feature": keeper,
                    "absolute_correlation": highest_correlation,
                    "removed_total_gain": float(total_gain.get(feature, 0.0)),
                    "kept_total_gain": float(total_gain.get(keeper, 0.0)),
                }
            )
        else:
            kept_features.append(feature)

    output = {
        "selection_source": "training months 0-59 only",
        "samples_per_month": SAMPLES_PER_MONTH,
        "sample_count": int(len(selected_indices)),
        "correlation_method": "Pearson absolute correlation",
        "correlation_threshold": CORRELATION_THRESHOLD,
        "original_feature_count": len(feature_columns),
        "kept_feature_count": len(kept_features),
        "removed_feature_count": len(removed),
        "kept_features_in_gain_order": kept_features,
        "removed_pairs": removed,
    }
    output_path = (
        project_dir
        / "data"
        / "interim"
        / "exp053r_correlated_feature_selection.json"
    )
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"sample_count={len(selected_indices)}, kept={len(kept_features)}, "
        f"removed={len(removed)}",
        flush=True,
    )
    for pair in removed:
        print(
            f"drop {pair['removed_feature']} -> keep {pair['kept_feature']} "
            f"corr={pair['absolute_correlation']:.6f}",
            flush=True,
        )
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
