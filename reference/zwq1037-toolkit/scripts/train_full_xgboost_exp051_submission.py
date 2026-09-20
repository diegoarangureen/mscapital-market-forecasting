"""Train leakage-safer EXP051 on months 0--70 and create its Kaggle submission."""

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
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters
from train_full_xgboost_exp034_submission import add_selected_event_tables
from train_full_xgboost_submissions import load_features


EXPERIMENT_ID = "EXP-TREE-051"
OUTPUT_NAME = "xgboost_exp051_rfmf_safe_fulltrain"
CHUNK_QUANTILE_FEATURES = {
    "t_large_buy_90",
    "t_large_sell_95",
    "x_large_trade_imbalance",
}


def selected_public_features(project_dir: Path) -> tuple[list[str], list[str]]:
    """Read the training-only overlap audit and apply EXP051's extra exclusions."""

    audit_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-050"
        / "public_feature_overlap_audit.csv"
    )
    audit = pd.read_csv(audit_path)
    audit_kept = audit.loc[
        audit["keep_for_incremental_test"].astype(bool), "public_feature"
    ].tolist()
    selected = [name for name in audit_kept if name not in CHUNK_QUANTILE_FEATURES]
    dropped = audit.loc[
        ~audit["keep_for_incremental_test"].astype(bool), "public_feature"
    ].tolist() + sorted(CHUNK_QUANTILE_FEATURES)
    if len(selected) != 140 or len(dropped) != 12:
        raise AssertionError("Unexpected EXP051 public feature selection.")
    return selected, dropped


def load_public_table(
    project_dir: Path, split: str, feature_columns: list[str]
) -> pd.DataFrame:
    """Load only the selected public CSV columns using compact dtypes."""

    path = (
        project_dir
        / "data"
        / "interim"
        / "public_features"
        / "rfmf_0726data"
        / f"{split}.csv"
    )
    usecols = ["sample_id", *feature_columns]
    if split == "train":
        usecols.append("target")
    dtypes = {name: np.float32 for name in feature_columns}
    dtypes["sample_id"] = np.int32
    if split == "train":
        dtypes["target"] = np.float32
    data = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    if data["sample_id"].nunique() != len(data):
        raise AssertionError(f"Duplicate sample IDs in public {split} table.")
    if np.isinf(data[feature_columns].to_numpy(copy=False)).any():
        raise AssertionError(f"Infinite values in public {split} features.")
    return data


def add_gap_table(project_dir: Path, split: str, data: pd.DataFrame) -> pd.DataFrame:
    """Join the four transaction event-gap features used by EXP037 and EXP051."""

    gap = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / f"{split}_transaction_event_gap_features.feather"
    )
    output = data.merge(gap, on="sample_id", how="left", validate="one_to_one")
    del gap
    gc.collect()
    return output


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    public_features, dropped_public_features = selected_public_features(project_dir)

    train_data, base_features = load_features(project_dir, "train")
    test_data, test_base_features = load_features(project_dir, "test")
    if base_features != test_base_features:
        raise AssertionError("Train and test base feature schemas differ.")
    train_data = add_gap_table(
        project_dir,
        "train",
        add_selected_event_tables(project_dir, "train", train_data),
    )
    test_data = add_gap_table(
        project_dir,
        "test",
        add_selected_event_tables(project_dir, "test", test_data),
    )

    public_train = load_public_table(project_dir, "train", public_features).rename(
        columns={"target": "public_target"}
    )
    public_test = load_public_table(project_dir, "test", public_features)
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    train_data = (
        train_data.merge(public_train, on="sample_id", how="left", validate="one_to_one")
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    test_data = (
        test_data.merge(public_test, on="sample_id", how="left", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    target_difference = np.max(
        np.abs(
            train_data["public_target"].to_numpy(dtype=np.float64)
            - train_data["target"].to_numpy(dtype=np.float64)
        )
    )
    if target_difference > 1e-5:
        raise AssertionError(
            f"Public target does not match the official label: {target_difference}."
        )
    train_data = train_data.drop(columns="public_target")
    del public_train, public_test, labels
    gc.collect()

    feature_columns = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
    )
    if len(feature_columns) != 299 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("EXP051 must contain exactly 299 unique features.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Identity, time, or target entered EXP051 features.")

    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if set(train_data["month"].unique()) != set(range(71)):
        raise AssertionError("Training data must contain labelled months 0 through 70.")
    if not np.array_equal(test_data["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test feature order differs from the submission template.")
    training_target_mean = float(train_data["target"].mean())
    centered_target = train_data["target"] - training_target_mean

    # 首次提交保持与已验证 EXP051 完全相同的 CPU 参数。
    # Keep the first submission identical to the validated CPU EXP051 setup.
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    model = XGBRegressor(**parameters)
    start_time = time.perf_counter()
    model.fit(train_data[feature_columns], centered_target)
    training_seconds = time.perf_counter() - start_time
    prediction = np.asarray(model.predict(test_data[feature_columns]), dtype=np.float64)
    if prediction.shape != (len(template),) or not np.isfinite(prediction).all():
        raise AssertionError("EXP051 generated invalid test predictions.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate IDs or missing values.")

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

    validation_config = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / EXPERIMENT_ID
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "training_months": "0-70 (all labelled rows)",
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "public_selected_feature_count": len(public_features),
        "public_dropped_features": dropped_public_features,
        "training_target_mean": training_target_mean,
        "target_centered_without_adding_mean_back": True,
        "validation_metrics": {
            "official_60_70": validation_config["overall_cosine"],
            "official_62_70_without_66": validation_config[
                "cosine_62_70_without_66"
            ],
            "official_67_70": validation_config["cosine_67_70"],
            "monthly_population_std": validation_config["monthly_stability"][
                "monthly_population_std"
            ],
        },
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
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{EXPERIMENT_ID}: features={len(feature_columns)}, "
        f"train_rows={len(train_data)}, test_rows={len(test_data)}, "
        f"train_seconds={training_seconds:.2f}",
        flush=True,
    )
    print(f"submission={submission_path}", flush=True)
    print(f"prediction_summary={metadata['prediction_summary']}", flush=True)


if __name__ == "__main__":
    main()
