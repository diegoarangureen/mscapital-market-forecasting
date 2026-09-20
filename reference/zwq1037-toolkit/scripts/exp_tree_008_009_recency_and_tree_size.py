"""EXP-TREE-008/009: recency weighting and larger-tree checks on 97 features."""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


TREE_007_COSINE = 0.10939156482320557
VARIANTS = {
    "baseline": {
        "experiment_id": "INTERNAL-BASELINE",
        "max_leaf_nodes": 31,
        "use_recency_weight": False,
    },
    "recency_weight_1_to_2": {
        "experiment_id": "EXP-TREE-008",
        "max_leaf_nodes": 31,
        "use_recency_weight": True,
    },
    "max_leaf_nodes_63": {
        "experiment_id": "EXP-TREE-009",
        "max_leaf_nodes": 63,
        "use_recency_weight": False,
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
    """Return cross-month level and volatility diagnostics."""

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


def build_recency_weights(training_months):
    """Map the earliest training month to 1 and latest to 2."""

    month_values = np.asarray(training_months, dtype=np.float64)
    minimum_month = float(month_values.min())
    maximum_month = float(month_values.max())
    if maximum_month == minimum_month:
        return np.ones_like(month_values)
    return 1.0 + (month_values - minimum_month) / (maximum_month - minimum_month)


def fit_variant(model_data, feature_columns, train_condition, valid_condition, variant_name):
    """Fit one controlled variant and return model, predictions, and metrics."""

    variant = VARIANTS[variant_name]
    X_train = model_data.loc[train_condition, feature_columns].copy()
    X_valid = model_data.loc[valid_condition, feature_columns].copy()
    y_train = model_data.loc[train_condition, "target"].copy()
    training_months = model_data.loc[train_condition, "month"].to_numpy()
    validation_output = model_data.loc[
        valid_condition, ["sample_id", "month", "target"]
    ].copy()

    # 加权实验必须同时使用加权target均值，避免额外引入常数方向偏移。
    # Weighted training uses the weighted target mean to avoid a second intercept change.
    if variant["use_recency_weight"]:
        sample_weight = build_recency_weights(training_months)
        training_target_mean = float(
            np.average(y_train.to_numpy(dtype=np.float64), weights=sample_weight)
        )
    else:
        sample_weight = None
        training_target_mean = float(y_train.mean())
    y_train_centered = y_train - training_target_mean

    model_parameters = {
        "loss": "squared_error",
        "learning_rate": 0.03,
        "max_iter": 400,
        "max_leaf_nodes": variant["max_leaf_nodes"],
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
        "verbose": 0,
    }
    model = HistGradientBoostingRegressor(**model_parameters)
    start_time = time.perf_counter()
    model.fit(X_train, y_train_centered, sample_weight=sample_weight)
    training_seconds = time.perf_counter() - start_time
    validation_predictions = model.predict(X_valid)
    if not np.isfinite(validation_predictions).all():
        raise AssertionError(f"{variant_name} produced non-finite predictions.")
    validation_output["prediction"] = validation_predictions
    monthly_scores = make_monthly_scores(validation_output)
    metadata = {
        "variant": variant_name,
        "experiment_id": variant["experiment_id"],
        "train_month_first": int(training_months.min()),
        "train_month_last": int(training_months.max()),
        "validation_month_first": int(validation_output["month"].min()),
        "validation_month_last": int(validation_output["month"].max()),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "training_target_mean": training_target_mean,
        "weight_min": float(sample_weight.min()) if sample_weight is not None else 1.0,
        "weight_max": float(sample_weight.max()) if sample_weight is not None else 1.0,
        "model_parameters": model_parameters,
        "training_seconds": training_seconds,
        "overall_cosine": cosine_similarity_score(
            validation_output["target"], validation_output["prediction"]
        ),
        "monthly_stability": summarize_monthly_stability(monthly_scores),
    }
    del X_train, X_valid, y_train, y_train_centered
    gc.collect()
    return model, validation_output, monthly_scores, metadata


def make_monthly_comparison(baseline_monthly, candidate_monthly):
    """Pair candidate and baseline month scores using identical month keys."""

    comparison = baseline_monthly[["month", "row_count", "cosine"]].rename(
        columns={"cosine": "baseline_cosine"}
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
        comparison["candidate_cosine"] - comparison["baseline_cosine"]
    )
    return comparison


def main():
    project_dir = Path(__file__).resolve().parents[1]

    # 读取与EXP-TREE-007完全相同的97个特征。
    # Load exactly the same 97 features as EXP-TREE-007.
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
        raise AssertionError("Expected 97 unique feature columns.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Forbidden identity, time, or target columns entered X.")
    del v1_data, last60_data, level2_data, label_data
    gc.collect()

    # 先在两个更早的forward fold中同时训练基线与两个候选。
    # Screen baseline and both candidates on two earlier forward folds first.
    fold_specs = [
        ("train_0_39_valid_40_49", 39, 40, 49),
        ("train_0_49_valid_50_59", 49, 50, 59),
    ]
    internal_summary_rows = []
    internal_prediction_parts = []
    for fold_name, train_last, valid_first, valid_last in fold_specs:
        train_condition = model_data["month"] <= train_last
        valid_condition = model_data["month"].between(valid_first, valid_last)
        for variant_name in VARIANTS:
            model, predictions, monthly_scores, metadata = fit_variant(
                model_data,
                feature_columns,
                train_condition,
                valid_condition,
                variant_name,
            )
            del model
            internal_summary_rows.append(
                {
                    "fold": fold_name,
                    "variant": variant_name,
                    "overall_cosine": metadata["overall_cosine"],
                    **metadata["monthly_stability"],
                    "training_seconds": metadata["training_seconds"],
                }
            )
            predictions["fold"] = fold_name
            predictions["variant"] = variant_name
            internal_prediction_parts.append(predictions)
            print(
                f"finished {fold_name} / {variant_name}: "
                f"cosine={metadata['overall_cosine']:.10f}",
                flush=True,
            )
            del predictions, monthly_scores
            gc.collect()

    internal_summary = pd.DataFrame(internal_summary_rows)
    internal_predictions = pd.concat(internal_prediction_parts, ignore_index=True)
    shared_run_dir = (
        project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-008-009"
    )
    shared_run_dir.mkdir(parents=True, exist_ok=True)
    internal_summary.to_csv(
        shared_run_dir / "internal_forward_summary.csv", index=False, encoding="utf-8"
    )
    internal_predictions.to_feather(
        shared_run_dir / "internal_forward_predictions.feather"
    )

    # 两个候选分别在固定0～59→60～70上正式运行；彼此不叠加。
    # Run both candidates separately on the fixed formal split; do not combine them.
    formal_train = model_data["month"] <= 59
    formal_valid = model_data["month"] >= 60
    baseline_monthly = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-007"
        / "monthly_cosine.csv"
    )
    baseline_stability = summarize_monthly_stability(baseline_monthly)
    for variant_name in ["recency_weight_1_to_2", "max_leaf_nodes_63"]:
        variant = VARIANTS[variant_name]
        experiment_id = variant["experiment_id"]
        run_dir = (
            project_dir / "data" / "interim" / "tree_experiments" / experiment_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        model, predictions, monthly_scores, metadata = fit_variant(
            model_data,
            feature_columns,
            formal_train,
            formal_valid,
            variant_name,
        )
        monthly_comparison = make_monthly_comparison(
            baseline_monthly, monthly_scores
        )
        prediction_path = (
            project_dir
            / "outputs"
            / "predictions"
            / f"hist_gradient_boosting_level2_{variant_name}_valid.feather"
        )
        model_path = (
            project_dir
            / "outputs"
            / "models"
            / f"hist_gradient_boosting_level2_{variant_name}.joblib"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_feather(prediction_path)
        monthly_scores.to_csv(
            run_dir / "monthly_cosine.csv", index=False, encoding="utf-8"
        )
        monthly_comparison.to_csv(
            run_dir / "monthly_comparison.csv", index=False, encoding="utf-8"
        )
        joblib.dump(model, model_path)

        candidate_stability = metadata["monthly_stability"]
        internal_variant_rows = internal_summary.loc[
            internal_summary["variant"].isin(["baseline", variant_name])
        ].to_dict(orient="records")
        config = {
            "experiment_id": experiment_id,
            "baseline": "EXP-TREE-007",
            "main_change": (
                "linear month sample weights from 1 to 2 with weighted target centering"
                if variant_name == "recency_weight_1_to_2"
                else "max_leaf_nodes from 31 to 63"
            ),
            "feature_count": len(feature_columns),
            "train_months": "0-59",
            "validation_months": "60-70",
            "formal_metadata": metadata,
            "exp_tree_007_cosine": TREE_007_COSINE,
            "absolute_change_from_exp_tree_007": (
                metadata["overall_cosine"] - TREE_007_COSINE
            ),
            "baseline_monthly_stability": baseline_stability,
            "monthly_stability_change": {
                key: candidate_stability[key] - baseline_stability[key]
                for key in [
                    "monthly_macro_mean",
                    "monthly_population_std",
                    "monthly_range",
                    "negative_month_count",
                ]
            },
            "internal_forward_results": internal_variant_rows,
            "model_path": str(model_path),
            "prediction_path": str(prediction_path),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"formal {experiment_id} / {variant_name}: "
            f"cosine={metadata['overall_cosine']:.10f}",
            flush=True,
        )
        del model, predictions, monthly_scores
        gc.collect()


if __name__ == "__main__":
    main()
