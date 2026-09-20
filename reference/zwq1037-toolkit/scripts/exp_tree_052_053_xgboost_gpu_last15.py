"""Run a clean GPU control and then add the compact final-15-second block."""

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
    make_monthly_scores,
)
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import (
    add_event_features,
    evaluate_and_save,
    fit_model,
    model_parameters,
    subset_cosine,
)
from exp_tree_051_xgboost_no_chunk_quantile_features import CHUNK_QUANTILE_FEATURES
from train_full_xgboost_submissions import load_features


def finalize_metadata(
    *,
    project_dir: Path,
    experiment_id: str,
    metadata: dict,
    baseline_predictions: pd.DataFrame,
    public_features: list[str],
    dropped_public_features: list[str],
) -> None:
    """Record authoritative persisted scores and the established robustness gates."""

    persisted = pd.read_feather(metadata["prediction_path"])
    metadata["overall_cosine"] = subset_cosine(persisted, "month >= 60")
    metadata["cosine_62_70"] = subset_cosine(persisted, "month >= 62")
    metadata["cosine_62_70_without_66"] = subset_cosine(
        persisted, "month >= 62 and month != 66"
    )
    metadata["cosine_67_70"] = subset_cosine(persisted, "month >= 67")
    metadata["cosine_60_64"] = subset_cosine(persisted, "month <= 64")

    baseline_recent = subset_cosine(baseline_predictions, "month >= 67")
    baseline_early = subset_cosine(baseline_predictions, "month <= 64")
    baseline_without_66 = subset_cosine(
        baseline_predictions, "month >= 62 and month != 66"
    )
    metadata["passes_core_generalization_gates"] = bool(
        metadata["cosine_62_70_without_66"] > baseline_without_66
        and metadata["cosine_67_70"] > baseline_recent
        and metadata["cosine_60_64"] >= baseline_early - 0.001
        and metadata["primary_leave_one_month_out_min_change"] >= 0
    )
    baseline_monthly = make_monthly_scores(baseline_predictions)
    baseline_primary = baseline_monthly.query("month >= 62 and month != 66")
    metadata["passes_strict_stability_gates"] = bool(
        metadata["passes_core_generalization_gates"]
        and metadata["primary_monthly_worst"] >= baseline_primary["cosine"].min()
        and metadata["primary_monthly_q25"]
        >= baseline_primary["cosine"].quantile(0.25)
    )
    metadata["public_source_feature_count"] = 152
    metadata["public_selected_feature_count"] = len(public_features)
    metadata["public_dropped_feature_count"] = len(dropped_public_features)
    metadata["public_dropped_features"] = dropped_public_features
    metadata["extra_excluded_public_features"] = list(CHUNK_QUANTILE_FEATURES)
    metadata["training_device"] = "cuda"
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_one(
    *,
    project_dir: Path,
    experiment_id: str,
    description: str,
    model_data: pd.DataFrame,
    feature_columns: list[str],
    train_mask: pd.Series,
    valid_mask: pd.Series,
    centered_target: pd.Series,
    validation_rows: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_id: str,
    public_features: list[str],
    dropped_public_features: list[str],
) -> pd.DataFrame:
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    parameters["device"] = "cuda"
    model, training_seconds = fit_model(
        model_data,
        feature_columns,
        train_mask,
        centered_target,
        parameters,
    )
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=experiment_id,
        description=description,
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id=baseline_id,
    )
    finalize_metadata(
        project_dir=project_dir,
        experiment_id=experiment_id,
        metadata=metadata,
        baseline_predictions=baseline_predictions,
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )
    del model
    gc.collect()
    return pd.read_feather(metadata["prediction_path"])


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    audit_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-050"
        / "public_feature_overlap_audit.csv"
    )
    audit = pd.read_csv(audit_path)
    audit_approved = audit.loc[
        audit["keep_for_incremental_test"].astype(bool), "public_feature"
    ].tolist()
    public_features = [
        name for name in audit_approved if name not in CHUNK_QUANTILE_FEATURES
    ]
    dropped_public_features = audit.loc[
        ~audit["keep_for_incremental_test"].astype(bool), "public_feature"
    ].tolist() + list(CHUNK_QUANTILE_FEATURES)
    if len(public_features) != 140 or len(dropped_public_features) != 12:
        raise AssertionError("Unexpected EXP051 public feature selection.")

    public_csv = (
        project_dir
        / "data"
        / "interim"
        / "public_features"
        / "rfmf_0726data"
        / "train.csv"
    )
    dtypes = {name: np.float32 for name in public_features + ["target"]}
    dtypes["sample_id"] = np.int32
    public = pd.read_csv(
        public_csv,
        usecols=["sample_id", *public_features, "target"],
        dtype=dtypes,
    ).rename(columns={"target": "public_target"})

    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    gap = pd.read_feather(
        project_dir / "data" / "processed" / "train_transaction_event_gap_features.feather"
    )
    last15 = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_last15_baseline_features.feather"
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
    del public, gap, last15, labels
    gc.collect()

    exp037_features = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
    )
    control_features = exp037_features + public_features
    candidate_features = control_features + LAST15_FEATURE_COLUMNS
    if len(control_features) != 299 or len(candidate_features) != 307:
        raise AssertionError("Unexpected GPU experiment feature count.")
    if len(candidate_features) != len(set(candidate_features)):
        raise AssertionError("Duplicate columns entered the candidate model.")

    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    exp051_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-051_valid.feather"
    )
    assert_prediction_alignment(exp051_predictions, validation_rows)

    print("training EXP-TREE-052 GPU control", flush=True)
    exp052_predictions = run_one(
        project_dir=project_dir,
        experiment_id="EXP-TREE-052",
        description="EXP051 feature schema trained with XGBoost CUDA as a GPU control",
        model_data=model_data,
        feature_columns=control_features,
        train_mask=train_mask,
        valid_mask=valid_mask,
        centered_target=centered_target,
        validation_rows=validation_rows,
        baseline_predictions=exp051_predictions,
        baseline_id="EXP-TREE-051",
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )
    print("training EXP-TREE-053 with final-15-second market block", flush=True)
    run_one(
        project_dir=project_dir,
        experiment_id="EXP-TREE-053",
        description=(
            "EXP052 GPU control plus seven public-baseline final-15-second market "
            "statistics and final-15/full realized-volatility ratio"
        ),
        model_data=model_data,
        feature_columns=candidate_features,
        train_mask=train_mask,
        valid_mask=valid_mask,
        centered_target=centered_target,
        validation_rows=validation_rows,
        baseline_predictions=exp052_predictions,
        baseline_id="EXP-TREE-052",
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )


if __name__ == "__main__":
    main()
