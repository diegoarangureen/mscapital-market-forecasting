"""EXP-TREE-003/004: one-parameter LightGBM comparisons after EXP-TREE-002.

Both variants preserve the target-centering transformation that won EXP-TREE-002.
Select exactly one named variant per run; its directory contains all evidence
needed to compare it fairly against EXP-TREE-002.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


TREE_002_COSINE = 0.10535858341911877
RANDOM_SEED = 42
VARIANTS = {
    "num_leaves_63": {
        "experiment_id": "EXP-TREE-003",
        "parameter_name": "num_leaves",
        "parameter_value": 63,
        "hypothesis": "More leaves may capture additional nonlinear interactions in the fixed 62 features.",
    },
    "min_child_samples_200": {
        "experiment_id": "EXP-TREE-004",
        "parameter_name": "min_child_samples",
        "parameter_value": 200,
        "hypothesis": "Larger leaves may reduce noise fitting and improve future-month stability.",
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


def lightgbm_cosine_metric(y_true, y_pred):
    """Use cosine only as whole-validation-vector evaluation, never as per-row loss."""

    return "cosine", cosine_similarity_score(y_true, y_pred), True


def make_monthly_scores(validation_output):
    """Calculate official cosine independently for each held-out month."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / variant["experiment_id"]
    )
    model_path = (
        project_dir
        / "outputs"
        / "models"
        / f"lightgbm_market_v1_last60_target_centered_{args.variant}.txt"
    )
    prediction_path = (
        project_dir
        / "outputs"
        / "predictions"
        / f"lightgbm_market_v1_last60_target_centered_{args.variant}_valid.feather"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    # 完整V1表作为左表，保持 EXP-002 的5个空窗口样本处理规则。
    # Keep V1 as the left master table, including EXP-002's empty-window samples.
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
        raise AssertionError("Forbidden identity, time, or target columns entered X.")

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
        raise AssertionError("Validation sample IDs must be unique.")
    training_target_mean = float(y_train.mean())
    y_train_centered = y_train - training_target_mean
    del v1_data, last60_data, label_data, model_data
    gc.collect()

    # 除这个单参数外，模型与 EXP-TREE-002 / EXP-002 完全一致。
    # Apart from this one parameter, match EXP-TREE-002 and EXP-002 exactly.
    model_parameters = {
        "objective": "regression",
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "metric": "None",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model_parameters[variant["parameter_name"]] = variant["parameter_value"]
    model = LGBMRegressor(**model_parameters)
    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train_centered,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_names=["forward_valid"],
        eval_metric=lightgbm_cosine_metric,
        callbacks=[
            early_stopping(stopping_rounds=200, first_metric_only=True, verbose=True),
            log_evaluation(period=100),
        ],
    )
    training_seconds = time.perf_counter() - start_time

    validation_predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
    if not np.isfinite(validation_predictions).all():
        raise AssertionError("Validation predictions contain non-finite values.")
    validation_output["prediction"] = validation_predictions
    overall_cosine = cosine_similarity_score(y_valid, validation_predictions)
    monthly_scores = make_monthly_scores(validation_output)

    model_path.write_text(
        model.booster_.model_to_string(num_iteration=model.best_iteration_), encoding="utf-8"
    )
    validation_output.to_feather(prediction_path)
    monthly_scores.to_csv(run_dir / "monthly_cosine.csv", index=False, encoding="utf-8")
    config = {
        "experiment_id": variant["experiment_id"],
        "baseline": "EXP-TREE-002",
        "main_change": f"{variant['parameter_name']}={variant['parameter_value']}",
        "hypothesis": variant["hypothesis"],
        "feature_set": "EXP-002 62 features",
        "feature_count": len(feature_columns),
        "train_months": "0-59",
        "validation_months": "60-70",
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "training_target_mean": training_target_mean,
        "model_parameters": model_parameters,
        "best_iteration": int(model.best_iteration_),
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
