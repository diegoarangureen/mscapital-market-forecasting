"""EXP-TREE-007: add mirrored level-two book features to the best tree model."""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


TREE_005_COSINE = 0.10618231448470118


def cosine_similarity_score(y_true, y_pred):
    """Compute the official uncentered whole-vector cosine score."""

    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(true_values, predicted_values) / denominator)


def make_monthly_scores(validation_output):
    """Calculate official cosine independently for every validation month."""

    records = []
    for month, month_data in validation_output.groupby("month", sort=True):
        records.append(
            {
                "month": int(month),
                "row_count": int(len(month_data)),
                "cosine": cosine_similarity_score(
                    month_data["target"], month_data["prediction"]
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_monthly_stability(monthly_scores):
    """Summarize cross-month level and volatility without replacing overall cosine."""

    cosine_values = monthly_scores["cosine"].to_numpy(dtype=np.float64)
    worst_position = int(np.argmin(cosine_values))
    best_position = int(np.argmax(cosine_values))
    return {
        "monthly_macro_mean": float(cosine_values.mean()),
        "monthly_population_std": float(cosine_values.std(ddof=0)),
        "monthly_range": float(cosine_values.max() - cosine_values.min()),
        "worst_month": int(monthly_scores.iloc[worst_position]["month"]),
        "worst_month_cosine": float(cosine_values[worst_position]),
        "best_month": int(monthly_scores.iloc[best_position]["month"]),
        "best_month_cosine": float(cosine_values[best_position]),
        "negative_month_count": int((cosine_values < 0.0).sum()),
    }


def main():
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-007"
    model_path = (
        project_dir
        / "outputs"
        / "models"
        / "hist_gradient_boosting_target_centered_level2.joblib"
    )
    prediction_path = (
        project_dir
        / "outputs"
        / "predictions"
        / "hist_gradient_boosting_target_centered_level2_valid.feather"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    # 三张特征表只按唯一sample_id连接；V1继续作为完整主表。
    # Join three feature tables only by unique sample ID, retaining V1 as the master.
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
    base_feature_columns = v1_data.columns[1:].tolist() + last60_data.columns[1:].tolist()
    level2_feature_columns = level2_data.columns[1:].tolist()
    if set(base_feature_columns).intersection(level2_feature_columns):
        raise AssertionError("Level-two feature names overlap the existing 62 features.")
    feature_columns = base_feature_columns + level2_feature_columns
    if len(feature_columns) != 97:
        raise AssertionError(f"Expected 97 total features, got {len(feature_columns)}.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Forbidden identity, time, or target columns entered X.")

    # 固定月份前向验证，不引入随机划分。
    # Keep the fixed forward split without any random validation split.
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
    if not validation_output["sample_id"].is_unique:
        raise AssertionError("Validation sample IDs must remain unique.")

    # 沿用当前最佳树模型的训练期target去均值规则。
    # Reuse the best tree model's training-only target-centering rule.
    training_target_mean = float(y_train.mean())
    y_train_centered = y_train - training_target_mean
    del v1_data, last60_data, level2_data, label_data, model_data
    gc.collect()

    # 模型参数与 EXP-TREE-005 完全相同；唯一主要变量是新增第二档特征。
    # Match EXP-TREE-005 exactly; the level-two feature group is the sole main change.
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
    monthly_scores = make_monthly_scores(validation_output)
    monthly_stability = summarize_monthly_stability(monthly_scores)

    # 用同一套定义复算旧基线的月度波动，避免比较口径不一致。
    # Recompute baseline stability with the same definitions.
    baseline_monthly_scores = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-005"
        / "monthly_cosine.csv"
    )
    baseline_monthly_stability = summarize_monthly_stability(baseline_monthly_scores)
    monthly_comparison = baseline_monthly_scores[["month", "row_count", "cosine"]].rename(
        columns={"cosine": "baseline_cosine"}
    )
    monthly_comparison = monthly_comparison.merge(
        monthly_scores[["month", "cosine"]].rename(
            columns={"cosine": "level2_cosine"}
        ),
        on="month",
        how="inner",
        validate="one_to_one",
    )
    monthly_comparison["change"] = (
        monthly_comparison["level2_cosine"]
        - monthly_comparison["baseline_cosine"]
    )
    stability_change = {
        "monthly_macro_mean_change": (
            monthly_stability["monthly_macro_mean"]
            - baseline_monthly_stability["monthly_macro_mean"]
        ),
        "monthly_population_std_change": (
            monthly_stability["monthly_population_std"]
            - baseline_monthly_stability["monthly_population_std"]
        ),
        "monthly_range_change": (
            monthly_stability["monthly_range"]
            - baseline_monthly_stability["monthly_range"]
        ),
        "negative_month_count_change": (
            monthly_stability["negative_month_count"]
            - baseline_monthly_stability["negative_month_count"]
        ),
    }

    joblib.dump(model, model_path)
    validation_output.to_feather(prediction_path)
    monthly_scores.to_csv(run_dir / "monthly_cosine.csv", index=False, encoding="utf-8")
    monthly_comparison.to_csv(
        run_dir / "monthly_comparison.csv", index=False, encoding="utf-8"
    )
    config = {
        "experiment_id": "EXP-TREE-007",
        "baseline": "EXP-TREE-005",
        "main_change": "add 35 mirrored full-window and last60 level-two order-book features",
        "base_feature_count": len(base_feature_columns),
        "new_level2_feature_count": len(level2_feature_columns),
        "total_feature_count": len(feature_columns),
        "level2_feature_columns": level2_feature_columns,
        "train_months": "0-59",
        "validation_months": "60-70",
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "training_target_mean": training_target_mean,
        "model_parameters": model_parameters,
        "training_seconds": training_seconds,
        "overall_cosine": overall_cosine,
        "exp_tree_005_cosine": TREE_005_COSINE,
        "absolute_change_from_exp_tree_005": overall_cosine - TREE_005_COSINE,
        "monthly_stability": monthly_stability,
        "baseline_monthly_stability": baseline_monthly_stability,
        "monthly_stability_change": stability_change,
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
