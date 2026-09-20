"""Controlled XGBoost tests for depth and two financial feature families."""

from __future__ import annotations

import gc
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from build_train_market_microstructure_features import (
    BOOK_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
)


VARIANTS = {
    "EXP-TREE-014": {
        "name": "depth_6",
        "description": "97 features; increase max_depth from 5 to 6 only",
        "max_depth": 6,
        "extra_features": [],
    },
    "EXP-TREE-015": {
        "name": "price_dynamics",
        "description": "depth 5; add returns, volatility, rolling moments, momentum and spread",
        "max_depth": 5,
        "extra_features": PRICE_FEATURE_COLUMNS,
    },
    "EXP-TREE-016": {
        "name": "book_microstructure",
        "description": "depth 5; add robust imbalance, OFI, OFI acceleration and microprice",
        "max_depth": 5,
        "extra_features": BOOK_FEATURE_COLUMNS,
    },
}


def cosine_similarity_score(y_true, y_pred):
    """Compute the official uncentered whole-vector cosine score."""

    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(true_values, predicted_values) / denominator)


def make_monthly_scores(prediction_data):
    """Calculate official cosine independently for every validation month."""

    rows = []
    for month, month_data in prediction_data.groupby("month", sort=True):
        rows.append(
            {
                "month": int(month),
                "row_count": int(len(month_data)),
                "cosine": cosine_similarity_score(
                    month_data["target"], month_data["prediction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_monthly_stability(monthly_scores):
    """Summarize month-level score variation alongside the overall score."""

    values = monthly_scores["cosine"].to_numpy(dtype=np.float64)
    worst_position = int(np.argmin(values))
    best_position = int(np.argmax(values))
    return {
        "monthly_macro_mean": float(values.mean()),
        "monthly_population_std": float(values.std(ddof=0)),
        "monthly_range": float(values.max() - values.min()),
        "worst_month": int(monthly_scores.iloc[worst_position]["month"]),
        "worst_month_cosine": float(values[worst_position]),
        "best_month": int(monthly_scores.iloc[best_position]["month"]),
        "best_month_cosine": float(values[best_position]),
        "negative_month_count": int((values < 0.0).sum()),
    }


def make_model(max_depth):
    """Create the controlled XGBoost configuration."""

    parameters = {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": max_depth,
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


def fit_one(model_data, feature_columns, train_condition, valid_condition, max_depth):
    """Fit on an earlier time span and predict only the later span."""

    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    prediction_data = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()

    # 只用训练月份的target均值做中心化，验证标签不参与任何训练变换。
    # Center with the training-month target mean; validation labels never affect training.
    training_target_mean = float(y_train.mean())
    y_train_centered = y_train - training_target_mean
    model, parameters = make_model(max_depth)
    start_time = time.perf_counter()
    model.fit(X_train, y_train_centered)
    training_seconds = time.perf_counter() - start_time
    predictions = np.asarray(model.predict(X_valid), dtype=np.float64).reshape(-1)
    if not np.isfinite(predictions).all():
        raise AssertionError("XGBoost produced non-finite predictions.")
    prediction_data["prediction"] = predictions
    monthly_scores = make_monthly_scores(prediction_data)
    metadata = {
        "training_target_mean": training_target_mean,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "training_seconds": training_seconds,
        "parameters": parameters,
        "overall_cosine": cosine_similarity_score(
            prediction_data["target"], prediction_data["prediction"]
        ),
        "monthly_stability": summarize_monthly_stability(monthly_scores),
    }
    del X_train, X_valid, y_train, y_train_centered
    gc.collect()
    return model, prediction_data, monthly_scores, metadata


def compare_months(baseline_predictions, candidate_monthly):
    """Pair candidate month scores with the depth-5 97-feature baseline."""

    baseline_monthly = make_monthly_scores(baseline_predictions)
    comparison = baseline_monthly.rename(
        columns={"cosine": "baseline_cosine"}
    )[["month", "row_count", "baseline_cosine"]]
    comparison = comparison.merge(
        candidate_monthly[["month", "cosine"]].rename(
            columns={"cosine": "candidate_cosine"}
        ),
        on="month",
        how="inner",
        validate="one_to_one",
    )
    comparison["change"] = comparison["candidate_cosine"] - comparison["baseline_cosine"]
    return comparison


def save_model_safely(model, output_path):
    """Save an XGBoost JSON model through an ASCII temporary path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        model.save_model(str(temporary_path))
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def assert_prediction_alignment(reference, candidate):
    """Ensure comparisons use exactly the same samples, months, and targets."""

    for column_name in ["sample_id", "month", "target"]:
        if not np.array_equal(reference[column_name].to_numpy(), candidate[column_name].to_numpy()):
            raise AssertionError(f"Prediction alignment mismatch: {column_name}")


def parse_arguments():
    """Allow a resumable run of one explicitly named controlled experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-id",
        choices=sorted(VARIANTS),
        help="Run only one experiment instead of the full controlled group.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parents[1]
    v1_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_features.feather"
    )
    last60_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_last60_features_complete.feather"
    )
    level2_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_level2_features.feather"
    )
    microstructure_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_microstructure_features.feather"
    )
    label_data = pd.read_feather(project_dir / "data" / "raw" / "label.feather")

    base_feature_columns = (
        v1_data.columns[1:].tolist()
        + last60_data.columns[1:].tolist()
        + level2_data.columns[1:].tolist()
    )
    if len(base_feature_columns) != 97 or len(set(base_feature_columns)) != 97:
        raise AssertionError("Expected 97 unique baseline features.")
    if microstructure_data.columns[1:].tolist() != PRICE_FEATURE_COLUMNS + BOOK_FEATURE_COLUMNS:
        raise AssertionError("Microstructure feature schema mismatch.")

    model_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(level2_data, on="sample_id", how="left", validate="one_to_one")
        .merge(microstructure_data, on="sample_id", how="left", validate="one_to_one")
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)
    del v1_data, last60_data, level2_data, microstructure_data, label_data
    gc.collect()

    forbidden = {"sample_id", "month", "target"}
    if forbidden.intersection(base_feature_columns + PRICE_FEATURE_COLUMNS + BOOK_FEATURE_COLUMNS):
        raise AssertionError("Identity, month, or target entered the feature list.")

    baseline_internal = pd.read_feather(
        project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-011" / "internal_predictions.feather"
    )
    baseline_formal = pd.read_feather(
        project_dir / "outputs" / "predictions" / "xgboost_target_centered_level2_valid.feather"
    )
    baseline_config = json.loads(
        (project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-011" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_internal_score = baseline_config["internal_metadata"]["overall_cosine"]
    baseline_formal_score = baseline_config["formal_metadata"]["overall_cosine"]
    baseline_internal_stability = baseline_config["internal_metadata"]["monthly_stability"]
    baseline_formal_stability = baseline_config["formal_metadata"]["monthly_stability"]

    internal_train = model_data["month"] <= 49
    internal_valid = model_data["month"].between(50, 59)
    formal_train = model_data["month"] <= 59
    formal_valid = model_data["month"] >= 60

    selected_variants = VARIANTS
    if arguments.experiment_id is not None:
        selected_variants = {
            arguments.experiment_id: VARIANTS[arguments.experiment_id]
        }

    for experiment_id, spec in selected_variants.items():
        feature_columns = base_feature_columns + spec["extra_features"]
        if len(feature_columns) != len(set(feature_columns)):
            raise AssertionError(f"Duplicate features in {experiment_id}.")
        run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)

        internal_model, internal_predictions, internal_monthly, internal_metadata = fit_one(
            model_data,
            feature_columns,
            internal_train,
            internal_valid,
            spec["max_depth"],
        )
        del internal_model
        assert_prediction_alignment(baseline_internal, internal_predictions)
        internal_comparison = compare_months(baseline_internal, internal_monthly)
        internal_predictions.to_feather(run_dir / "internal_predictions.feather")
        internal_monthly.to_csv(run_dir / "internal_monthly_cosine.csv", index=False)
        internal_comparison.to_csv(run_dir / "internal_monthly_comparison.csv", index=False)
        print(
            f"{experiment_id} internal: cosine={internal_metadata['overall_cosine']:.10f}, "
            f"change={internal_metadata['overall_cosine'] - baseline_internal_score:+.10f}",
            flush=True,
        )
        del internal_predictions, internal_monthly, internal_comparison
        gc.collect()

        formal_model, formal_predictions, formal_monthly, formal_metadata = fit_one(
            model_data,
            feature_columns,
            formal_train,
            formal_valid,
            spec["max_depth"],
        )
        assert_prediction_alignment(baseline_formal, formal_predictions)
        formal_comparison = compare_months(baseline_formal, formal_monthly)
        prediction_path = (
            project_dir / "outputs" / "predictions" / f"xgboost_{spec['name']}_valid.feather"
        )
        model_path = project_dir / "outputs" / "models" / f"xgboost_{spec['name']}.json"
        formal_predictions.to_feather(prediction_path)
        formal_monthly.to_csv(run_dir / "monthly_cosine.csv", index=False)
        formal_comparison.to_csv(run_dir / "monthly_comparison.csv", index=False)
        save_model_safely(formal_model, model_path)

        config = {
            "experiment_id": experiment_id,
            "description": spec["description"],
            "baseline": "EXP-TREE-011 XGBoost depth 5 with 97 features",
            "main_change": (
                "max_depth 5 to 6 only"
                if experiment_id == "EXP-TREE-014"
                else f"add {len(spec['extra_features'])} {spec['name']} features only"
            ),
            "feature_count": len(feature_columns),
            "added_feature_columns": spec["extra_features"],
            "train_months": "0-59",
            "validation_months": "60-70",
            "internal_split": "train 0-49, validate 50-59",
            "xgboost_version": xgboost.__version__,
            "internal_metadata": internal_metadata,
            "internal_baseline_cosine": baseline_internal_score,
            "internal_change_from_baseline": internal_metadata["overall_cosine"] - baseline_internal_score,
            "internal_baseline_monthly_stability": baseline_internal_stability,
            "formal_metadata": formal_metadata,
            "formal_baseline_cosine": baseline_formal_score,
            "formal_change_from_baseline": formal_metadata["overall_cosine"] - baseline_formal_score,
            "formal_baseline_monthly_stability": baseline_formal_stability,
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
            f"change={formal_metadata['overall_cosine'] - baseline_formal_score:+.10f}, "
            f"monthly_std={formal_metadata['monthly_stability']['monthly_population_std']:.10f}",
            flush=True,
        )
        del formal_model, formal_predictions, formal_monthly, formal_comparison
        gc.collect()


if __name__ == "__main__":
    main()
