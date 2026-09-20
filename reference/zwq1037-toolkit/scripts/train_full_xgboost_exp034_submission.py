"""Train EXP-TREE-034 on months 0--70 and create its Kaggle submission."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters
from train_full_xgboost_submissions import load_features


EXPERIMENT_ID = "EXP-TREE-034"
OUTPUT_NAME = "xgboost_exp034_fulltrain"


def add_selected_event_tables(
    project_dir: Path, split: str, data: pd.DataFrame
) -> pd.DataFrame:
    """Join transaction flow and multi-window order flow without the legacy order group."""

    processed_dir = project_dir / "data" / "processed"
    result = data
    for table_name in [
        f"{split}_transaction_flow_features.feather",
        f"{split}_order_multiwindow_features.feather",
    ]:
        table = pd.read_feather(processed_dir / table_name)
        if table["sample_id"].nunique() != len(table):
            raise AssertionError(f"Duplicate sample ids in {table_name}.")
        result = result.merge(table, on="sample_id", how="left", validate="one_to_one")
        del table
    gc.collect()
    return result


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    train_data, base_features = load_features(project_dir, "train")
    test_data, test_base_features = load_features(project_dir, "test")
    if base_features != test_base_features:
        raise AssertionError("Train and test base feature schemas differ.")
    train_data = add_selected_event_tables(project_dir, "train", train_data)
    test_data = add_selected_event_tables(project_dir, "test", test_data)

    feature_columns = base_features + TRANSACTION_FEATURE_COLUMNS + ORDER_MULTI_FEATURE_COLUMNS
    if len(feature_columns) != 155 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("EXP-TREE-034 must contain exactly 155 unique features.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Identity, time, or target entered the model features.")

    labels = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    train_data = train_data.merge(labels, on="sample_id", how="inner", validate="one_to_one")
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if set(train_data["month"].unique()) != set(range(71)):
        raise AssertionError("Training data must contain labelled months 0 through 70.")
    if not np.array_equal(test_data["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test feature order differs from the submission template.")

    training_target_mean = float(train_data["target"].mean())
    centered_target = train_data["target"] - training_target_mean
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    model = XGBRegressor(**parameters)
    start_time = time.perf_counter()
    model.fit(train_data[feature_columns], centered_target)
    training_seconds = time.perf_counter() - start_time
    prediction = np.asarray(model.predict(test_data[feature_columns]), dtype=np.float64)
    if prediction.shape != (len(template),) or not np.isfinite(prediction).all():
        raise AssertionError("EXP-TREE-034 generated invalid test predictions.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate ids or missing values.")

    submission_dir = project_dir / "outputs" / "submissions"
    model_dir = project_dir / "outputs" / "models"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    for output_dir in [submission_dir, model_dir, metadata_dir]:
        output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{OUTPUT_NAME}.csv"
    model_path = model_dir / f"{OUTPUT_NAME}.json"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    submission.to_csv(submission_path, index=False)
    save_model_safely(model, model_path)

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "training_months": "0-70 (all labelled rows)",
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": len(feature_columns),
        "feature_groups": ["base132", "transaction8", "order_multiwindow15"],
        "feature_columns": feature_columns,
        "training_target_mean": training_target_mean,
        "target_centered_without_adding_mean_back": True,
        "validation_cosine_60_70": 0.1414142821,
        "robustness_cosine_62_70": 0.1427449099,
        "robustness_cosine_62_70_without_66": 0.1282196703,
        "validation_monthly_population_std": 0.0175282339,
        "parameters": parameters,
        "xgboost_version": xgboost.__version__,
        "training_seconds": training_seconds,
        "prediction_summary": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std(ddof=0)),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
        },
        "submission_path": str(submission_path),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{EXPERIMENT_ID}: features={len(feature_columns)}, train_rows={len(train_data)}, "
        f"test_rows={len(test_data)}, train_seconds={training_seconds:.2f}",
        flush=True,
    )
    print(f"submission={submission_path}", flush=True)
    print(f"prediction_summary={metadata['prediction_summary']}", flush=True)


if __name__ == "__main__":
    main()
