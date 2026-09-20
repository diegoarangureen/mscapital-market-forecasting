"""EXP-TREE-005: sklearn HistGradientBoosting on the EXP-TREE-002 protocol."""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


TREE_002_COSINE = 0.10535858341911877


def cosine_similarity_score(y_true, y_pred):
    """Compute the official uncentered whole-vector cosine score."""

    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(true_values, predicted_values) / denominator)


def main():
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-005"
    model_path = project_dir / "outputs" / "models" / "hist_gradient_boosting_target_centered.joblib"
    prediction_path = (
        project_dir
        / "outputs"
        / "predictions"
        / "hist_gradient_boosting_target_centered_valid.feather"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    # 特征、left join 和空窗口规则与 EXP-002 完全相同。
    # Feature set, left join, and empty-window handling exactly match EXP-002.
    v1_data = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_features.feather"
    )
    last60_data = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / "train_market_last60_features_complete.feather"
    )
    label_data = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    model_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)
    feature_columns = v1_data.columns[1:].tolist() + last60_data.columns[1:].tolist()
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Forbidden columns entered the feature matrix.")
    train_condition = model_data["month"] <= 59
    valid_condition = model_data["month"] >= 60
    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    y_valid = model_data.loc[valid_condition, "target"].copy()
    validation_output = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()
    if (len(X_train), len(X_valid)) != (1_064_163, 193_474):
        raise AssertionError("The fixed forward split changed unexpectedly.")

    # 只用训练期 target 均值；与现有最强树模型的 target 方向一致。
    # Use only the training target mean, matching the strongest current tree direction.
    training_target_mean = float(y_train.mean())
    y_train_centered = y_train - training_target_mean
    del v1_data, last60_data, label_data, model_data
    gc.collect()

    # 禁用 sklearn 内部随机 validation；400轮是预先固定预算，不从60～70月挑轮数。
    # Disable sklearn's random validation; 400 is a fixed budget, not validation tuning.
    model_parameters = {
        "loss": "squared_error",
        "learning_rate": 0.03,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
        "verbose": 0,
    }
    model = HistGradientBoostingRegressor(**model_parameters)
    start_time = time.perf_counter()
    model.fit(X_train, y_train_centered)
    training_seconds = time.perf_counter() - start_time
    validation_predictions = model.predict(X_valid)
    if not np.isfinite(validation_predictions).all():
        raise AssertionError("Validation predictions contain non-finite values.")
    validation_output["prediction"] = validation_predictions
    overall_cosine = cosine_similarity_score(y_valid, validation_predictions)

    monthly_rows = []
    for month, month_data in validation_output.groupby("month", sort=True):
        monthly_rows.append(
            {
                "month": int(month),
                "row_count": int(len(month_data)),
                "cosine": cosine_similarity_score(
                    month_data["target"], month_data["prediction"]
                ),
            }
        )
    monthly_scores = pd.DataFrame(monthly_rows)
    joblib.dump(model, model_path)
    validation_output.to_feather(prediction_path)
    monthly_scores.to_csv(run_dir / "monthly_cosine.csv", index=False, encoding="utf-8")
    config = {
        "experiment_id": "EXP-TREE-005",
        "baseline": "EXP-TREE-002",
        "main_change": "replace LightGBM with sklearn HistGradientBoostingRegressor",
        "feature_set": "EXP-002 62 features",
        "feature_count": len(feature_columns),
        "train_months": "0-59",
        "validation_months": "60-70",
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "training_target_mean": training_target_mean,
        "model_parameters": model_parameters,
        "training_seconds": training_seconds,
        "overall_cosine": overall_cosine,
        "exp_tree_002_cosine": TREE_002_COSINE,
        "absolute_change_from_exp_tree_002": overall_cosine - TREE_002_COSINE,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(monthly_scores.to_string(index=False))


if __name__ == "__main__":
    main()
