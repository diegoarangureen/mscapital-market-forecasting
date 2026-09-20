"""EXP-TREE-001: small, reproducible diagnostics for cosine-aligned tree training.

This script is deliberately a research screen, not a leaderboard experiment.  It
uses every tenth sample ID from the fixed forward split, preserves all months,
and writes its configuration, predictions, overall scores, and per-month scores
to a separate traditional-tree output directory.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


RANDOM_SEED = 42
TRAIN_MONTH_LAST = 59
VALID_MONTH_FIRST = 60


def cosine_similarity_score(y_true, y_pred):
    """Return the official whole-vector cosine score with a zero-vector guard."""

    # 统一为一维 float64，避免大向量点积的额外数值误差。
    # Use one-dimensional float64 arrays for stable large-vector dot products.
    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(true_values, predicted_values) / denominator)


def lightgbm_cosine_metric(y_true, y_pred):
    """Adapt the official vector metric to LightGBM evaluation only."""

    return "cosine", cosine_similarity_score(y_true, y_pred), True


def negative_cosine_objective(y_true, y_pred):
    """A diagnostic-only diagonal approximation to negative whole-vector cosine.

    This is mathematically the gradient of ``-cos(y, p)`` with respect to every
    prediction p_i.  The true Hessian has off-diagonal terms, because changing
    one prediction changes the norm of the entire prediction vector.  LightGBM
    only accepts one Hessian value per row, so this function clips the diagonal
    after dropping those cross-row terms.  It is therefore an experiment, not a
    claim that LightGBM exactly optimizes official cosine.
    """

    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true_norm = np.linalg.norm(true_values)
    prediction_norm = np.linalg.norm(predicted_values)

    # 初始常数预测可能接近零；此处只防止除零，不改变后续公式。
    # Initial constant predictions can be near zero; guard only the denominator.
    safe_prediction_norm = max(prediction_norm, 1e-12)
    safe_true_norm = max(true_norm, 1e-12)
    cosine_value = np.dot(true_values, predicted_values) / (
        safe_true_norm * safe_prediction_norm
    )

    # gradient = d[-cos(y, p)] / d p_i。
    # The gradient is the exact global-vector derivative for each prediction.
    gradient = (
        -true_values / (safe_true_norm * safe_prediction_norm)
        + cosine_value * predicted_values / safe_prediction_norm**2
    )

    # 对角二阶项；完整 Hessian 还包含 i != j 的耦合项，LightGBM 无法接收。
    # Diagonal Hessian only; the full Hessian also has unavailable cross-row terms.
    diagonal_hessian = (
        true_values * predicted_values
        / (safe_true_norm * safe_prediction_norm**3)
        + cosine_value / safe_prediction_norm**2
        - 3.0 * cosine_value * predicted_values**2 / safe_prediction_norm**4
    )

    # 树的 Newton 叶子计算要求正的二阶权重；负值显示该近似并不天然稳健。
    # Tree Newton updates require positive curvature; clipping exposes the limitation.
    hessian = np.maximum(diagonal_hessian, 1e-6)
    return gradient, hessian


def make_model(objective="regression"):
    """Create the EXP-002 parameter set; objective is the sole model-side change."""

    return LGBMRegressor(
        objective=objective,
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        metric="None",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_model(
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    sample_weight=None,
):
    """Fit one model and return its validation predictions and reproducible metadata."""

    model = make_model(
        negative_cosine_objective if model_name == "custom_negative_cosine" else "regression"
    )
    stopping_callback = early_stopping(
        stopping_rounds=100,
        first_metric_only=True,
        verbose=False,
    )
    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_names=["forward_valid"],
        eval_metric=lightgbm_cosine_metric,
        callbacks=[stopping_callback, log_evaluation(period=0)],
    )
    predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{model_name} produced non-finite validation predictions.")

    metadata = {
        "best_iteration": int(model.best_iteration_),
        "training_seconds": time.perf_counter() - start_time,
    }
    return model, np.asarray(predictions, dtype=np.float64), metadata


def fit_oof_predictions(X_train, y_train, months):
    """Make time-respecting OOF predictions for a one-parameter intercept calibrator."""

    # 两个 calibration 块都只由更早月份训练，避免本块 target 泄漏进其预测。
    # Each calibration block is predicted from earlier months only.
    fold_specs = [(39, 40, 49), (49, 50, 59)]
    oof_parts = []
    for train_last, calibration_first, calibration_last in fold_specs:
        fold_train = months <= train_last
        fold_calibration = (months >= calibration_first) & (months <= calibration_last)
        fold_model, fold_predictions, fold_metadata = fit_model(
            "l2_oof_fold",
            X_train.loc[fold_train],
            y_train.loc[fold_train],
            X_train.loc[fold_calibration],
            y_train.loc[fold_calibration],
        )
        del fold_model
        gc.collect()
        oof_parts.append(
            pd.DataFrame(
                {
                    "target": y_train.loc[fold_calibration].to_numpy(),
                    "prediction": fold_predictions,
                    "fold_train_last_month": train_last,
                    "fold_best_iteration": fold_metadata["best_iteration"],
                }
            )
        )
    return pd.concat(oof_parts, ignore_index=True)


def best_cosine_intercept(targets, predictions):
    """Fit b in p + b*1 that maximizes cosine on a labelled calibration vector.

    Positive scaling is redundant for cosine, so an affine calibration has only
    one effective direction parameter: the constant-vector shift b.
    """

    true_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    dot_y_p = float(np.dot(true_values, predicted_values))
    sum_y = float(true_values.sum())
    norm_p_squared = float(np.dot(predicted_values, predicted_values))
    sum_p = float(predicted_values.sum())
    sample_count = float(len(true_values))

    # 对 cos(y, p+b1) 求导可得到此驻点；随后直接比较候选分数以防符号误判。
    # Differentiate cos(y, p+b1), then score candidates to avoid sign mistakes.
    denominator = sum_y * sum_p - dot_y_p * sample_count
    if abs(denominator) < 1e-18:
        return 0.0
    stationary_b = (dot_y_p * sum_p - sum_y * norm_p_squared) / denominator
    candidate_values = [0.0, stationary_b]
    candidate_scores = [
        cosine_similarity_score(true_values, predicted_values + candidate_b)
        for candidate_b in candidate_values
    ]
    return float(candidate_values[int(np.argmax(candidate_scores))])


def score_predictions(prediction_table, prediction_columns):
    """Return overall and month-by-month official cosine for each named prediction."""

    scores = {"overall": {}, "monthly": {}}
    for column_name in prediction_columns:
        scores["overall"][column_name] = cosine_similarity_score(
            prediction_table["target"], prediction_table[column_name]
        )
        monthly_scores = {}
        for month, month_data in prediction_table.groupby("month", sort=True):
            monthly_scores[str(int(month))] = cosine_similarity_score(
                month_data["target"], month_data[column_name]
            )
        scores["monthly"][column_name] = monthly_scores
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-modulus",
        type=int,
        default=10,
        help="Keep rows whose sample_id modulo this value is zero (default: 10).",
    )
    parser.add_argument(
        "--sample-remainder",
        type=int,
        default=0,
        help="Remainder retained by the deterministic sample rule (default: 0).",
    )
    args = parser.parse_args()
    if args.sample_modulus < 2:
        raise ValueError("sample_modulus must be at least 2 for a small research screen.")
    if not 0 <= args.sample_remainder < args.sample_modulus:
        raise ValueError("sample_remainder must be in [0, sample_modulus).")

    project_dir = Path(__file__).resolve().parents[1]
    output_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_cosine_research"
        / f"sample_mod_{args.sample_modulus}_remainder_{args.sample_remainder}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # V1 完整表作左表，确保最后60秒缺失样本也在研究样本中。
    # Keep V1 as the master table so empty last-60-second windows remain included.
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
    sample_condition = (
        model_data["sample_id"] % args.sample_modulus == args.sample_remainder
    )
    model_data = model_data.loc[sample_condition].copy()

    train_condition = model_data["month"] <= TRAIN_MONTH_LAST
    valid_condition = model_data["month"] >= VALID_MONTH_FIRST
    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    y_valid = model_data.loc[valid_condition, "target"].copy()
    train_months = model_data.loc[train_condition, "month"].copy()
    prediction_table = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()
    del v1_data, last60_data, label_data, model_data
    gc.collect()

    # L2 is the direct EXP-002 model-family control on this exact small split.
    # L2 是与 EXP-002 同族的受控对照。
    l2_model, raw_predictions, l2_metadata = fit_model(
        "l2_raw", X_train, y_train, X_valid, y_valid
    )
    prediction_table["l2_raw"] = raw_predictions
    prediction_table["l2_prediction_demeaned"] = raw_predictions - raw_predictions.mean()

    # 训练 target 去均值：仅使用训练期均值，且不把均值加回预测。
    # Center targets with the training mean only and intentionally do not add it back.
    training_target_mean = float(y_train.mean())
    centered_model, centered_predictions, centered_metadata = fit_model(
        "l2_target_centered",
        X_train,
        y_train - training_target_mean,
        X_valid,
        y_valid,
    )
    prediction_table["l2_target_centered"] = centered_predictions

    # 两阶段路线：OOF 预测拟合一个常数平移，再应用到未见过的60～70月。
    # Two-stage route: fit a constant shift on OOF predictions, then apply it to months 60-70.
    oof_data = fit_oof_predictions(X_train, y_train, train_months)
    oof_intercept = best_cosine_intercept(oof_data["target"], oof_data["prediction"])
    prediction_table["l2_oof_intercept"] = raw_predictions + oof_intercept

    # 样本权重只是可分解 L2 的代理，不是假装成逐样本 cosine。
    # Magnitude weights are a decomposable L2 proxy, not a pretend per-row cosine loss.
    absolute_target_weights = np.abs(y_train.to_numpy(dtype=np.float64))
    absolute_target_weights = absolute_target_weights / absolute_target_weights.mean()
    weighted_model, weighted_predictions, weighted_metadata = fit_model(
        "l2_abs_target_weight",
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weight=absolute_target_weights,
    )
    prediction_table["l2_abs_target_weight"] = weighted_predictions

    # 这是受限接口下的自定义目标诊断；结果无论好坏都不能消除全局 Hessian 耦合问题。
    # This custom-objective diagnostic cannot remove the unavailable global Hessian coupling.
    custom_status = "completed"
    custom_metadata = {}
    try:
        custom_model, custom_predictions, custom_metadata = fit_model(
            "custom_negative_cosine", X_train, y_train, X_valid, y_valid
        )
        prediction_table["custom_negative_cosine"] = custom_predictions
    except Exception as error:  # The failure itself is evidence about interface reliability.
        custom_status = f"failed: {type(error).__name__}: {error}"
        custom_model = None

    prediction_columns = [
        "l2_raw",
        "l2_prediction_demeaned",
        "l2_target_centered",
        "l2_oof_intercept",
        "l2_abs_target_weight",
    ]
    if custom_model is not None:
        prediction_columns.append("custom_negative_cosine")
    scores = score_predictions(prediction_table, prediction_columns)
    scores["diagnostics"] = {
        "positive_scale_x7_score": cosine_similarity_score(y_valid, 7.0 * raw_predictions),
        "validation_target_informed_best_intercept": best_cosine_intercept(
            y_valid, raw_predictions
        ),
        "validation_target_informed_best_intercept_score": cosine_similarity_score(
            y_valid,
            raw_predictions + best_cosine_intercept(y_valid, raw_predictions),
        ),
        "note": "The target-informed intercept is an upper-bound diagnostic only and is not a valid validation result.",
    }
    config = {
        "experiment_id": "EXP-TREE-001",
        "purpose": "small cosine-alignment research screen; not a formal full-data result",
        "feature_set": "EXP-002 market V1 plus last60, 62 features",
        "sample_rule": (
            f"sample_id % {args.sample_modulus} == {args.sample_remainder}"
        ),
        "train_months": "0-59",
        "validation_months": "60-70",
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "random_seed": RANDOM_SEED,
        "l2_metadata": l2_metadata,
        "target_centered_metadata": centered_metadata,
        "weighted_metadata": weighted_metadata,
        "custom_objective_status": custom_status,
        "custom_objective_metadata": custom_metadata,
        "oof_intercept": oof_intercept,
        "oof_rows": int(len(oof_data)),
        "training_target_mean": training_target_mean,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    oof_data.to_feather(output_dir / "oof_calibration_predictions.feather")
    prediction_table.to_feather(output_dir / "validation_predictions.feather")
    for model_name, model in {
        "l2_raw": l2_model,
        "l2_target_centered": centered_model,
        "l2_abs_target_weight": weighted_model,
        "custom_negative_cosine": custom_model,
    }.items():
        if model is not None:
            (output_dir / f"{model_name}.txt").write_text(
                model.booster_.model_to_string(num_iteration=model.best_iteration_),
                encoding="utf-8",
            )

    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
