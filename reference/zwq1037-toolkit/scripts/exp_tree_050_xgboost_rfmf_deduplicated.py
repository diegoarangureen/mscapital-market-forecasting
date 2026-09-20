"""Add training-only deduplicated public RFMF-0726 features to EXP037."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
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
from train_full_xgboost_submissions import load_features


EXPERIMENT_ID = "EXP-TREE-050"


def run_experiment(
    *,
    experiment_id: str = EXPERIMENT_ID,
    extra_excluded_public_features: tuple[str, ...] = (),
    description: str | None = None,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    audit_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
        / "public_feature_overlap_audit.csv"
    )
    audit = pd.read_csv(audit_path)
    audit_approved_features = audit.loc[
        audit["keep_for_incremental_test"].astype(bool), "public_feature"
    ].tolist()
    unknown_exclusions = set(extra_excluded_public_features) - set(
        audit_approved_features
    )
    if unknown_exclusions:
        raise AssertionError(f"Unknown extra public exclusions: {unknown_exclusions}.")
    public_features = [
        name
        for name in audit_approved_features
        if name not in extra_excluded_public_features
    ]
    dropped_public_features = audit.loc[
        ~audit["keep_for_incremental_test"].astype(bool), "public_feature"
    ].tolist() + list(extra_excluded_public_features)
    if (
        len(public_features) != 143 - len(extra_excluded_public_features)
        or len(dropped_public_features) != 9 + len(extra_excluded_public_features)
    ):
        raise AssertionError("Unexpected public-feature deduplication result.")

    # 只读取审计保留的列，避免把已判定重复的列再次载入训练矩阵。
    # Read only audit-approved columns so rejected duplicates never enter the model matrix.
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
    if len(public) != 1_257_637 or public["sample_id"].nunique() != len(public):
        raise AssertionError(f"Unexpected public feature table shape: {public.shape}.")
    if np.isinf(public[public_features].to_numpy(copy=False)).any():
        raise AssertionError("Selected public features contain infinity.")

    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    gap_features = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / "train_transaction_event_gap_features.feather"
    )
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    model_data = (
        model_data.merge(gap_features, on="sample_id", how="left", validate="one_to_one")
        .merge(public, on="sample_id", how="left", validate="one_to_one")
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
        raise AssertionError(
            f"Public target does not match the official label: {target_difference}."
        )
    model_data = model_data.drop(columns="public_target")
    del public, gap_features, labels
    gc.collect()

    exp037_features = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
    )
    feature_columns = exp037_features + public_features
    expected_feature_count = 302 - len(extra_excluded_public_features)
    if len(exp037_features) != 159 or len(feature_columns) != expected_feature_count:
        raise AssertionError(f"Unexpected {experiment_id} feature count.")
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError(f"Duplicate column names entered {experiment_id}.")

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
        description=description
        or (
            "EXP037 plus 143 public RFMF-0726 features after training-only "
            "semantic and abs-correlation >= 0.995 deduplication"
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
    metadata["public_source_feature_count"] = 152
    metadata["public_selected_feature_count"] = len(public_features)
    metadata["public_dropped_feature_count"] = len(dropped_public_features)
    metadata["public_dropped_features"] = dropped_public_features
    metadata["deduplication_scope"] = (
        "month 0-59 only; sample_id modulo 10; manual same-formula removal plus "
        "absolute Pearson correlation >= 0.995"
    )
    metadata["overlap_audit_path"] = str(audit_path)
    metadata["extra_excluded_public_features"] = list(
        extra_excluded_public_features
    )
    # 重新读取落盘预测并覆盖关键分数，避免内存对象与持久化结果不一致。
    # Re-read persisted predictions so recorded validation scores are authoritative.
    persisted_predictions = pd.read_feather(metadata["prediction_path"])
    metadata["overall_cosine"] = subset_cosine(persisted_predictions, "month >= 60")
    metadata["cosine_62_70"] = subset_cosine(persisted_predictions, "month >= 62")
    metadata["cosine_62_70_without_66"] = subset_cosine(
        persisted_predictions, "month >= 62 and month != 66"
    )
    metadata["cosine_67_70"] = subset_cosine(persisted_predictions, "month >= 67")
    metadata["cosine_60_64"] = subset_cosine(persisted_predictions, "month <= 64")
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
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
