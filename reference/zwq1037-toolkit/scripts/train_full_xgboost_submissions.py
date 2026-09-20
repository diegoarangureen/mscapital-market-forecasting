"""Train selected XGBoost candidates on months 0--70 and create submissions."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from build_event_flow_features import ORDER_FEATURE_COLUMNS, TRANSACTION_FEATURE_COLUMNS
from build_event_market_cross_features import CROSS_FEATURE_COLUMNS
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely


CANDIDATES = {
    "EXP-TREE-020": {
        "name": "xgboost_transaction_flow_fulltrain",
        "feature_groups": ["transaction"],
        "validation_cosine_60_70": 0.13628635806561576,
        "robustness_cosine_62_70": 0.1377956879208879,
    },
    "EXP-TREE-022": {
        "name": "xgboost_transaction_and_order_flow_fulltrain",
        "feature_groups": ["transaction", "order"],
        "validation_cosine_60_70": 0.13497332369436088,
        "robustness_cosine_62_70": 0.1358181509847253,
    },
    "EXP-TREE-023": {
        "name": "xgboost_transaction_flow_depth_cross_fulltrain",
        "feature_groups": ["transaction", "cross"],
        "validation_cosine_60_70": 0.13827519276299102,
        "robustness_cosine_62_70": 0.1399797547021849,
    },
}


def load_features(project_dir: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    """Load all reusable groups and return the exact 132-feature base table."""

    processed = project_dir / "data" / "processed"
    v1 = pd.read_feather(processed / f"{split}_market_features.feather")
    recent = pd.read_feather(processed / f"{split}_market_last60_features_complete.feather")
    level2 = pd.read_feather(processed / f"{split}_market_level2_features.feather")
    micro = pd.read_feather(
        processed / f"{split}_market_microstructure_features.feather",
        columns=["sample_id", *BOOK_FEATURE_COLUMNS],
    )
    base_features = (
        pd.read_feather(processed / "train_market_features.feather", columns=None).columns[1:].tolist()
        + pd.read_feather(
            processed / "train_market_last60_features_complete.feather", columns=None
        ).columns[1:].tolist()
        + pd.read_feather(
            processed / "train_market_level2_features.feather", columns=None
        ).columns[1:].tolist()
        + BOOK_FEATURE_COLUMNS
    )
    if len(base_features) != 132 or len(set(base_features)) != 132:
        raise AssertionError("Expected the exact 132-feature base schema.")
    data = (
        v1.merge(recent, on="sample_id", how="left", validate="one_to_one")
        .merge(level2, on="sample_id", how="left", validate="one_to_one")
        .merge(micro, on="sample_id", how="left", validate="one_to_one")
    )
    data["last60_row_count"] = data["last60_row_count"].fillna(0)
    del v1, recent, level2, micro
    gc.collect()
    return data, base_features


def add_event_tables(project_dir: Path, split: str, data: pd.DataFrame) -> pd.DataFrame:
    """Join the three compact event-derived groups exactly once."""

    processed = project_dir / "data" / "processed"
    transaction = pd.read_feather(processed / f"{split}_transaction_flow_features.feather")
    order = pd.read_feather(processed / f"{split}_order_flow_features.feather")
    cross = pd.read_feather(processed / f"{split}_event_market_cross_features.feather")
    result = (
        data.merge(transaction, on="sample_id", how="left", validate="one_to_one")
        .merge(order, on="sample_id", how="left", validate="one_to_one")
        .merge(cross, on="sample_id", how="left", validate="one_to_one")
    )
    return result


def feature_list(base_features: list[str], groups: list[str]) -> list[str]:
    """Return the declared model columns in their training order."""

    output = list(base_features)
    if "transaction" in groups:
        output.extend(TRANSACTION_FEATURE_COLUMNS)
    if "order" in groups:
        output.extend(ORDER_FEATURE_COLUMNS)
    if "cross" in groups:
        output.extend(CROSS_FEATURE_COLUMNS)
    if len(output) != len(set(output)):
        raise AssertionError("Duplicate model feature columns.")
    return output


def make_model() -> tuple[XGBRegressor, dict]:
    """Return the unchanged selected validation configuration."""

    parameters = {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 100.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    return XGBRegressor(**parameters), parameters


def main() -> None:
    """Train all selected candidates and save model, CSV, and reproducibility metadata."""

    project_dir = Path(__file__).resolve().parents[1]
    train_data, base_features = load_features(project_dir, "train")
    test_data, test_base_features = load_features(project_dir, "test")
    if base_features != test_base_features:
        raise AssertionError("Train and test base feature order differs.")
    label = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    train_data = add_event_tables(project_dir, "train", train_data).merge(
        label, on="sample_id", how="inner", validate="one_to_one"
    )
    test_data = add_event_tables(project_dir, "test", test_data)
    if not np.array_equal(test_data["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test feature order does not match the submission template.")
    if not train_data["month"].between(0, 70).all():
        raise AssertionError("Full training data does not cover exactly the labelled period.")
    training_target_mean = float(train_data["target"].mean())
    centered_target = train_data["target"] - training_target_mean
    submission_dir = project_dir / "outputs" / "submissions"
    model_dir = project_dir / "outputs" / "models"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    submission_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for experiment_id, spec in CANDIDATES.items():
        columns = feature_list(base_features, spec["feature_groups"])
        model, parameters = make_model()
        start_time = time.perf_counter()
        model.fit(train_data[columns], centered_target)
        training_seconds = time.perf_counter() - start_time
        prediction = np.asarray(model.predict(test_data[columns]), dtype=np.float64)
        if len(prediction) != len(template) or not np.isfinite(prediction).all():
            raise AssertionError(f"Invalid test predictions for {experiment_id}.")
        submission = template[["sample_id"]].copy()
        submission["prediction"] = prediction
        submission_path = submission_dir / f"{spec['name']}.csv"
        model_path = model_dir / f"{spec['name']}.json"
        submission.to_csv(submission_path, index=False)
        save_model_safely(model, model_path)
        metadata = {
            "experiment_id": experiment_id,
            "source_validation_experiment": experiment_id,
            "training_months": "0-70 (all labelled rows)",
            "train_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
            "feature_count": len(columns),
            "feature_groups": spec["feature_groups"],
            "feature_columns": columns,
            "training_target_mean": training_target_mean,
            "target_centered_without_adding_mean_back": True,
            "validation_cosine_60_70": spec["validation_cosine_60_70"],
            "robustness_cosine_62_70": spec["robustness_cosine_62_70"],
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
        (metadata_dir / f"{spec['name']}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{experiment_id}: features={len(columns)}, train_seconds={training_seconds:.2f}, "
            f"submission={submission_path}",
            flush=True,
        )
        del model, prediction, submission
        gc.collect()


if __name__ == "__main__":
    main()
