"""EXP-TREE-010..013: four 97-feature tree-model family comparisons."""

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import catboost
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor


INTERNAL_HGB_COSINE = 0.09361549801717718
FORMAL_HGB_COSINE = 0.10939156482320557
MODEL_SPECS = {
    "lightgbm": {
        "experiment_id": "EXP-TREE-010",
        "description": "LightGBM, fixed 400-tree budget",
    },
    "xgboost": {
        "experiment_id": "EXP-TREE-011",
        "description": "XGBoost hist, fixed 400-tree budget",
    },
    "catboost": {
        "experiment_id": "EXP-TREE-012",
        "description": "CatBoost symmetric trees, fixed 400-tree budget",
    },
    "extra_trees": {
        "experiment_id": "EXP-TREE-013",
        "description": "ExtraTrees, small fixed 32-tree budget",
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
    """Calculate official cosine independently for each validation month."""

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
    """Summarize month-level score and volatility without replacing overall cosine."""

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


def make_model(model_name):
    """Create one predeclared, bounded model-family configuration."""

    if model_name == "lightgbm":
        parameters = {
            "objective": "regression",
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "reg_lambda": 1.0,
            "metric": "None",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        return LGBMRegressor(**parameters), parameters

    if model_name == "xgboost":
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

    if model_name == "catboost":
        parameters = {
            "loss_function": "RMSE",
            "iterations": 400,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 1.0,
            "bootstrap_type": "No",
            "random_strength": 0.0,
            "random_seed": 42,
            "thread_count": -1,
            "allow_writing_files": False,
            "verbose": False,
        }
        return CatBoostRegressor(**parameters), parameters

    if model_name == "extra_trees":
        parameters = {
            "n_estimators": 32,
            "criterion": "squared_error",
            "max_depth": 16,
            "min_samples_leaf": 100,
            "max_features": 1.0,
            "bootstrap": False,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
        }
        return ExtraTreesRegressor(**parameters), parameters

    raise ValueError(f"Unknown model: {model_name}")


def save_model_safely(model_name, model, output_path):
    """Save each native model format, using ASCII temporary paths when needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "lightgbm":
        output_path.write_text(model.booster_.model_to_string(), encoding="utf-8")
        return
    if model_name == "extra_trees":
        joblib.dump(model, output_path)
        return

    suffix = ".json" if model_name == "xgboost" else ".cbm"
    temporary_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        model.save_model(str(temporary_path))
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def fit_one(model_name, model_data, feature_columns, train_condition, valid_condition):
    """Fit one model on an earlier period and predict only the later period."""

    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    prediction_data = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()

    # 所有模型使用同一训练期target均值，不使用验证target做平移。
    # Every model uses the same training-only target mean; validation labels never shift it.
    training_target_mean = float(y_train.mean())
    y_train_centered = y_train - training_target_mean
    model, parameters = make_model(model_name)
    start_time = time.perf_counter()
    model.fit(X_train, y_train_centered)
    training_seconds = time.perf_counter() - start_time
    predictions = np.asarray(model.predict(X_valid), dtype=np.float64).reshape(-1)
    if not np.isfinite(predictions).all():
        raise AssertionError(f"{model_name} produced non-finite predictions.")
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


def make_monthly_comparison(baseline_monthly, candidate_monthly):
    """Pair baseline and candidate month scores."""

    comparison = baseline_monthly[["month", "row_count", "cosine"]].rename(
        columns={"cosine": "hist_gradient_boosting_cosine"}
    )
    comparison = comparison.merge(
        candidate_monthly[["month", "cosine"]].rename(
            columns={"cosine": "candidate_cosine"}
        ),
        on="month",
        how="inner",
        validate="one_to_one",
    )
    comparison["change"] = (
        comparison["candidate_cosine"]
        - comparison["hist_gradient_boosting_cosine"]
    )
    return comparison


def main():
    project_dir = Path(__file__).resolve().parents[1]

    # 复用EXP-TREE-007的完整97特征，不新增特征或预处理。
    # Reuse EXP-TREE-007's 97 features without new features or preprocessing.
    v1_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_features.feather"
    )
    last60_data = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / "train_market_last60_features_complete.feather"
    )
    level2_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_level2_features.feather"
    )
    label_data = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    model_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(level2_data, on="sample_id", how="left", validate="one_to_one")
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)
    feature_columns = (
        v1_data.columns[1:].tolist()
        + last60_data.columns[1:].tolist()
        + level2_data.columns[1:].tolist()
    )
    if len(feature_columns) != 97 or len(set(feature_columns)) != 97:
        raise AssertionError("Expected 97 unique features.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Forbidden identity, time, or target columns entered X.")
    del v1_data, last60_data, level2_data, label_data
    gc.collect()

    # 最新内部时间折的HGB预测已经存在，直接复用以减少重复训练。
    # Reuse the existing HGB prediction for the latest internal fold.
    internal_hgb_all = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-008-009"
        / "internal_forward_predictions.feather"
    )
    internal_hgb = internal_hgb_all.loc[
        (internal_hgb_all["fold"] == "train_0_49_valid_50_59")
        & (internal_hgb_all["variant"] == "baseline"),
        ["sample_id", "month", "target", "prediction"],
    ].copy()
    internal_hgb_monthly = make_monthly_scores(internal_hgb)
    formal_hgb = pd.read_feather(
        project_dir
        / "outputs"
        / "predictions"
        / "hist_gradient_boosting_target_centered_level2_valid.feather"
    )
    formal_hgb_monthly = make_monthly_scores(formal_hgb)
    del internal_hgb_all
    gc.collect()

    internal_train = model_data["month"] <= 49
    internal_valid = model_data["month"].between(50, 59)
    formal_train = model_data["month"] <= 59
    formal_valid = model_data["month"] >= 60
    package_versions = {
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "sklearn": sklearn.__version__,
    }

    for model_name, spec in MODEL_SPECS.items():
        experiment_id = spec["experiment_id"]
        run_dir = (
            project_dir / "data" / "interim" / "tree_experiments" / experiment_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        internal_model, internal_predictions, internal_monthly, internal_metadata = fit_one(
            model_name, model_data, feature_columns, internal_train, internal_valid
        )
        del internal_model
        internal_predictions.to_feather(run_dir / "internal_predictions.feather")
        internal_monthly.to_csv(
            run_dir / "internal_monthly_cosine.csv", index=False, encoding="utf-8"
        )
        make_monthly_comparison(internal_hgb_monthly, internal_monthly).to_csv(
            run_dir / "internal_monthly_comparison.csv", index=False, encoding="utf-8"
        )
        print(
            f"{experiment_id} internal {model_name}: "
            f"cosine={internal_metadata['overall_cosine']:.10f}",
            flush=True,
        )
        del internal_predictions, internal_monthly
        gc.collect()

        formal_model, formal_predictions, formal_monthly, formal_metadata = fit_one(
            model_name, model_data, feature_columns, formal_train, formal_valid
        )
        prediction_path = (
            project_dir
            / "outputs"
            / "predictions"
            / f"{model_name}_target_centered_level2_valid.feather"
        )
        model_suffix = {
            "lightgbm": ".txt",
            "xgboost": ".json",
            "catboost": ".cbm",
            "extra_trees": ".joblib",
        }[model_name]
        model_path = (
            project_dir
            / "outputs"
            / "models"
            / f"{model_name}_target_centered_level2{model_suffix}"
        )
        formal_predictions.to_feather(prediction_path)
        formal_monthly.to_csv(
            run_dir / "monthly_cosine.csv", index=False, encoding="utf-8"
        )
        formal_comparison = make_monthly_comparison(
            formal_hgb_monthly, formal_monthly
        )
        formal_comparison.to_csv(
            run_dir / "monthly_comparison.csv", index=False, encoding="utf-8"
        )
        save_model_safely(model_name, formal_model, model_path)
        config = {
            "experiment_id": experiment_id,
            "description": spec["description"],
            "baseline": "EXP-TREE-007 HistGradientBoosting",
            "main_change": f"replace HistGradientBoosting with {model_name}",
            "feature_count": len(feature_columns),
            "train_months": "0-59",
            "validation_months": "60-70",
            "package_versions": package_versions,
            "internal_split": "train 0-49, validate 50-59",
            "internal_metadata": internal_metadata,
            "internal_hgb_cosine": INTERNAL_HGB_COSINE,
            "internal_change_from_hgb": (
                internal_metadata["overall_cosine"] - INTERNAL_HGB_COSINE
            ),
            "formal_metadata": formal_metadata,
            "formal_hgb_cosine": FORMAL_HGB_COSINE,
            "formal_change_from_hgb": (
                formal_metadata["overall_cosine"] - FORMAL_HGB_COSINE
            ),
            "formal_improved_month_count": int(
                (formal_comparison["change"] > 0.0).sum()
            ),
            "formal_declined_month_count": int(
                (formal_comparison["change"] < 0.0).sum()
            ),
            "model_path": str(model_path),
            "prediction_path": str(prediction_path),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{experiment_id} formal {model_name}: "
            f"cosine={formal_metadata['overall_cosine']:.10f}",
            flush=True,
        )
        del formal_model, formal_predictions, formal_monthly
        gc.collect()


if __name__ == "__main__":
    main()
