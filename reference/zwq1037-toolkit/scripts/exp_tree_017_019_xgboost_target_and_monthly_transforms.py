"""Audit raw targets and transductive monthly rank/z-score features for XGBoost.

Monthly transformations use every *unlabelled* feature row inside its own month.
They are valid transductive validation features, but require known test-month
boundaries to reproduce at submission time.  They never use the target column.
"""

from __future__ import annotations

import argparse
import gc
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


TOP_FEATURE_COUNT = 12
BASELINE_EXPERIMENT_ID = "EXP-TREE-016"
VARIANTS = {
    "EXP-TREE-017": {
        "name": "microstructure_raw_target",
        "description": "132 features; raw target instead of training-mean-centered target",
        "target_centered": False,
        "transform": None,
    },
    "EXP-TREE-018": {
        "name": "microstructure_monthly_rank",
        "description": "132 features plus monthly percentile ranks for fold-selected top features",
        "target_centered": True,
        "transform": "rank",
    },
    "EXP-TREE-019": {
        "name": "microstructure_monthly_zscore",
        "description": "132 features plus monthly z-scores for fold-selected top features",
        "target_centered": True,
        "transform": "zscore",
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
    """Calculate the official cosine separately in each later validation month."""

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
    """Describe level and variation of monthly cosine without replacing the official score."""

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


def make_model():
    """Return exactly the current depth-5 XGBoost configuration."""

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


def choose_top_features(model_data, base_features, train_condition):
    """Select the top gain features with a model fitted on this fold's training rows only."""

    X_train = model_data.loc[train_condition, base_features]
    y_train = model_data.loc[train_condition, "target"]
    y_centered = y_train - float(y_train.mean())
    selector, _ = make_model()
    selector.fit(X_train, y_centered)
    gain = selector.get_booster().get_score(importance_type="gain")
    selected = sorted(base_features, key=lambda name: gain.get(name, 0.0), reverse=True)[
        :TOP_FEATURE_COUNT
    ]
    if len(selected) != TOP_FEATURE_COUNT or len(set(selected)) != TOP_FEATURE_COUNT:
        raise AssertionError("Feature-selection result is malformed.")
    selected_gain = {name: float(gain.get(name, 0.0)) for name in selected}
    del selector, X_train, y_train, y_centered
    gc.collect()
    return selected, selected_gain


def add_monthly_transform(model_data, selected_features, transform):
    """Add rank or z-score columns computed within each month without target access."""

    output = model_data.copy()
    new_columns = []
    for feature_name in selected_features:
        new_name = f"monthly_{transform}__{feature_name}"
        if transform == "rank":
            # pct=True maps the least value toward 0 and the greatest value to 1.
            # Ties use their average rank; missing source values remain missing.
            output[new_name] = output.groupby("month", sort=False)[feature_name].rank(
                pct=True, method="average"
            ).astype(np.float32)
        elif transform == "zscore":
            grouped = output.groupby("month", sort=False)[feature_name]
            monthly_mean = grouped.transform("mean")
            monthly_std = grouped.transform("std")
            output[new_name] = ((output[feature_name] - monthly_mean) / monthly_std).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unknown monthly transform: {transform}")
        new_columns.append(new_name)
    values = output[new_columns].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise AssertionError("Monthly transformation generated infinity.")
    return output, new_columns


def fit_one(
    model_data,
    feature_columns,
    train_condition,
    valid_condition,
    target_centered,
):
    """Fit with raw or training-mean-centered labels and predict a later time block."""

    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    prediction_data = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()
    training_target_mean = float(y_train.mean())
    fit_target = y_train - training_target_mean if target_centered else y_train
    model, parameters = make_model()
    start_time = time.perf_counter()
    model.fit(X_train, fit_target)
    training_seconds = time.perf_counter() - start_time
    predictions = np.asarray(model.predict(X_valid), dtype=np.float64).reshape(-1)
    if not np.isfinite(predictions).all():
        raise AssertionError("XGBoost produced non-finite predictions.")
    prediction_data["prediction"] = predictions
    monthly_scores = make_monthly_scores(prediction_data)
    metadata = {
        "target_centered": target_centered,
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
    del X_train, X_valid, y_train, fit_target
    gc.collect()
    return model, prediction_data, monthly_scores, metadata


def compare_months(baseline_predictions, candidate_monthly):
    """Compare all months against the fixed current-best model."""

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
    """Save XGBoost JSON through an ASCII temporary path for Windows reliability."""

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
    """Check that the baseline and candidate refer to the same held-out rows."""

    for column_name in ["sample_id", "month", "target"]:
        if not np.array_equal(reference[column_name].to_numpy(), candidate[column_name].to_numpy()):
            raise AssertionError(f"Prediction alignment mismatch: {column_name}")


def parse_arguments():
    """Select one controlled experiment when resuming an interrupted run."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", choices=sorted(VARIANTS))
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parents[1]
    v1_data = pd.read_feather(project_dir / "data" / "processed" / "train_market_features.feather")
    last60_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_last60_features_complete.feather"
    )
    level2_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_level2_features.feather"
    )
    micro_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_microstructure_features.feather"
    )
    label_data = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    # EXP-TREE-016 的132列只包含稳健盘口/OFI列，不包含已单独失败的21列价格动态特征。
    # The 132-column EXP-TREE-016 baseline excludes the separately failed 21 price-dynamics columns.
    base_features = (
        v1_data.columns[1:].tolist()
        + last60_data.columns[1:].tolist()
        + level2_data.columns[1:].tolist()
        + BOOK_FEATURE_COLUMNS
    )
    if len(base_features) != 132 or len(set(base_features)) != 132:
        raise AssertionError("Expected 132 unique base features.")
    expected_micro = PRICE_FEATURE_COLUMNS + BOOK_FEATURE_COLUMNS
    if micro_data.columns[1:].tolist() != expected_micro:
        raise AssertionError("Microstructure schema mismatch.")
    model_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(level2_data, on="sample_id", how="left", validate="one_to_one")
        .merge(
            micro_data[["sample_id", *BOOK_FEATURE_COLUMNS]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)
    del v1_data, last60_data, level2_data, micro_data, label_data
    gc.collect()

    forbidden = {"sample_id", "month", "target"}
    if forbidden.intersection(base_features):
        raise AssertionError("Identity, month, or target entered X.")
    baseline_internal = pd.read_feather(
        project_dir / "data" / "interim" / "tree_experiments" / BASELINE_EXPERIMENT_ID / "internal_predictions.feather"
    )
    baseline_formal = pd.read_feather(
        project_dir / "outputs" / "predictions" / "xgboost_book_microstructure_valid.feather"
    )
    baseline_config = json.loads(
        (project_dir / "data" / "interim" / "tree_experiments" / BASELINE_EXPERIMENT_ID / "config.json").read_text(encoding="utf-8")
    )
    internal_train = model_data["month"] <= 49
    internal_valid = model_data["month"].between(50, 59)
    formal_train = model_data["month"] <= 59
    formal_valid = model_data["month"] >= 60

    selected_variants = VARIANTS
    if arguments.experiment_id:
        selected_variants = {arguments.experiment_id: VARIANTS[arguments.experiment_id]}

    for experiment_id, spec in selected_variants.items():
        internal_data = model_data
        formal_data = model_data
        internal_added = []
        formal_added = []
        internal_selected = []
        formal_selected = []
        internal_gain = {}
        formal_gain = {}
        if spec["transform"] is not None:
            # Selection is fold-local: neither 50--59 nor 60--70 labels guide the feature set.
            internal_selected, internal_gain = choose_top_features(
                model_data, base_features, internal_train
            )
            formal_selected, formal_gain = choose_top_features(
                model_data, base_features, formal_train
            )
            internal_data, internal_added = add_monthly_transform(
                model_data, internal_selected, spec["transform"]
            )
            formal_data, formal_added = add_monthly_transform(
                model_data, formal_selected, spec["transform"]
            )

        run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        internal_features = base_features + internal_added
        formal_features = base_features + formal_added
        internal_model, internal_predictions, internal_monthly, internal_metadata = fit_one(
            internal_data, internal_features, internal_train, internal_valid, spec["target_centered"]
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
        del internal_predictions, internal_monthly, internal_comparison, internal_data
        gc.collect()

        formal_model, formal_predictions, formal_monthly, formal_metadata = fit_one(
            formal_data, formal_features, formal_train, formal_valid, spec["target_centered"]
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
            "baseline": "EXP-TREE-016 XGBoost, 132 features, target centered",
            "main_change": spec["description"],
            "formal_protocol": "train months 0-59; validate months 60-70; official uncentered whole-vector cosine",
            "internal_protocol": "train months 0-49; validate months 50-59",
            "transductive_monthly_features": spec["transform"] is not None,
            "transductive_note": (
                "Rank/z-score is calculated from all unlabelled rows inside each month. "
                "This cannot be reproduced on test unless month boundaries are available."
                if spec["transform"] is not None else None
            ),
            "base_feature_count": len(base_features),
            "internal_feature_count": len(internal_features),
            "formal_feature_count": len(formal_features),
            "internal_selected_features": internal_selected,
            "formal_selected_features": formal_selected,
            "internal_selected_gain": internal_gain,
            "formal_selected_gain": formal_gain,
            "internal_added_feature_columns": internal_added,
            "formal_added_feature_columns": formal_added,
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
        (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{experiment_id} formal: cosine={formal_metadata['overall_cosine']:.10f}, "
            f"change={formal_metadata['overall_cosine'] - baseline_config['formal_metadata']['overall_cosine']:+.10f}, "
            f"monthly_std={formal_metadata['monthly_stability']['monthly_population_std']:.10f}",
            flush=True,
        )
        del formal_model, formal_predictions, formal_monthly, formal_comparison, formal_data
        gc.collect()


if __name__ == "__main__":
    main()
