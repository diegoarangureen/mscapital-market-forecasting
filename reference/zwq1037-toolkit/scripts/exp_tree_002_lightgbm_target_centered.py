"""EXP-TREE-002: EXP-002 LightGBM with training-target centering only.

The competition score is an uncentered whole-vector cosine.  This script tests
one precise hypothesis: fitting L2 to ``target - train_target_mean`` and keeping
that centered prediction direction may better match cosine.  It never uses a
validation target to choose the transformation and does not change features,
the time split, LightGBM hyperparameters, or early-stopping rule from EXP-002.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


EXP_002_COSINE = 0.10387003918505122
RANDOM_SEED = 42


def cosine_similarity_score(y_true, y_pred):
    """Compute the official uncentered cosine with a zero-vector guard."""

    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(true_values, predicted_values) / denominator)


def lightgbm_cosine_metric(y_true, y_pred):
    """Expose the whole-validation-vector score as an evaluation metric only."""

    return "cosine", cosine_similarity_score(y_true, y_pred), True


def monthly_cosine_table(validation_output):
    """Calculate the same official cosine separately for each held-out month."""

    rows = []
    for month, month_data in validation_output.groupby("month", sort=True):
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


def main():
    project_dir = Path(__file__).resolve().parents[1]
    v1_path = project_dir / "data" / "processed" / "train_market_features.feather"
    last60_path = (
        project_dir
        / "data"
        / "processed"
        / "train_market_last60_features_complete.feather"
    )
    label_path = project_dir / "data" / "raw" / "label.feather"
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-002"
    model_path = (
        project_dir / "outputs" / "models" / "lightgbm_market_v1_last60_target_centered.txt"
    )
    prediction_path = (
        project_dir
        / "outputs"
        / "predictions"
        / "lightgbm_market_v1_last60_target_centered_valid.feather"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    # V1 是完整主表；left join 保留最后60秒没有记录的5个样本。
    # V1 is the complete master table; left join retains empty last-60-second windows.
    v1_data = pd.read_feather(v1_path)
    last60_data = pd.read_feather(last60_path)
    label_data = pd.read_feather(label_path)
    model_data = (
        v1_data.merge(last60_data, on="sample_id", how="left", validate="one_to_one")
        .merge(label_data, on="sample_id", how="inner", validate="one_to_one")
    )
    model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)
    if len(model_data) != len(v1_data):
        raise AssertionError("The complete V1 sample set was not preserved.")

    feature_columns = v1_data.columns[1:].tolist() + last60_data.columns[1:].tolist()
    forbidden_columns = {"sample_id", "month", "target"}
    if forbidden_columns.intersection(feature_columns):
        raise AssertionError("Identity, time, or target columns entered the feature matrix.")

    # 固定 forward validation；绝不随机划分。
    # Fixed forward validation only; never use a random split.
    train_condition = model_data["month"] <= 59
    valid_condition = model_data["month"] >= 60
    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    y_valid = model_data.loc[valid_condition, "target"].copy()
    validation_output = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()
    if len(X_train) != 1_064_163 or len(X_valid) != 193_474:
        raise AssertionError("Unexpected fixed split row counts.")
    if not validation_output["sample_id"].is_unique:
        raise AssertionError("Validation sample IDs must remain unique.")

    # 只由0～59月学习 target 均值；60～70月 target 从不参与此变换。
    # Learn the target mean from months 0-59 only; validation labels never set it.
    train_target_mean = float(y_train.mean())
    y_train_centered = y_train - train_target_mean
    del v1_data, last60_data, label_data, model_data
    gc.collect()

    # 除 target 的训练期平移外，所有 LightGBM 设置与 EXP-002 完全相同。
    # Apart from the training-target shift, every LightGBM setting matches EXP-002.
    model = LGBMRegressor(
        objective="regression",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        metric="None",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    stopping_callback = early_stopping(
        stopping_rounds=200,
        first_metric_only=True,
        verbose=True,
    )
    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train_centered,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_names=["forward_valid"],
        eval_metric=lightgbm_cosine_metric,
        callbacks=[stopping_callback, log_evaluation(period=100)],
    )
    training_seconds = time.perf_counter() - start_time

    # 不加回训练均值：我们检验的是 centered direction 对官方未中心化 cosine 的影响。
    # Do not add the training mean back: test the centered direction on official cosine.
    validation_predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
    if not np.isfinite(validation_predictions).all():
        raise AssertionError("Validation predictions contain non-finite values.")
    validation_output["prediction"] = validation_predictions
    overall_cosine = cosine_similarity_score(y_valid, validation_predictions)
    monthly_scores = monthly_cosine_table(validation_output)

    # Python 写文本绕过 LightGBM C++ 写入接口对中文路径的限制。
    # Use Python text output to avoid LightGBM C++ Unicode-path limitations.
    model_path.write_text(
        model.booster_.model_to_string(num_iteration=model.best_iteration_), encoding="utf-8"
    )
    prediction_output = validation_output.copy()
    prediction_output.to_feather(prediction_path)
    monthly_scores.to_csv(run_dir / "monthly_cosine.csv", index=False, encoding="utf-8")
    config = {
        "experiment_id": "EXP-TREE-002",
        "baseline": "EXP-002",
        "main_change": "fit L2 to target - mean(target_train_months_0_to_59); do not add mean back to predictions",
        "feature_count": len(feature_columns),
        "feature_set": "40 Market V1 plus 22 last60 features",
        "train_months": "0-59",
        "validation_months": "60-70",
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "train_target_mean": train_target_mean,
        "lightgbm_parameters": model.get_params(),
        "best_iteration": int(model.best_iteration_),
        "training_seconds": training_seconds,
        "overall_cosine": overall_cosine,
        "exp_002_cosine": EXP_002_COSINE,
        "absolute_change_from_exp_002": overall_cosine - EXP_002_COSINE,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(config, ensure_ascii=False, indent=2, default=str))
    print(monthly_scores.to_string(index=False))


if __name__ == "__main__":
    main()
