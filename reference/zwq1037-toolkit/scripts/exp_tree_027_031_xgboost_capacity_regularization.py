"""Controlled capacity, regularization, pruning, and cross-feature tests on EXP-TREE-025."""

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
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
    compare_months,
    cosine_similarity_score,
    make_monthly_scores,
    save_model_safely,
    summarize_monthly_stability,
)
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from train_full_xgboost_submissions import load_features


BASELINE_ID = "EXP-TREE-025"
CROSS_FEATURE_COLUMNS = [
    "trade_order_pressure_gap_60",
    "trade_order_pressure_product_60",
    "trade_order_pressure_gap_20",
    "trade_order_pressure_product_20",
]


def add_event_features(project_dir: Path, data: pd.DataFrame) -> pd.DataFrame:
    """Join the exact event tables used by EXP-TREE-025."""

    processed_dir = project_dir / "data" / "processed"
    result = data
    for table_name in [
        "train_transaction_flow_features.feather",
        "train_order_flow_features.feather",
        "train_order_multiwindow_features.feather",
    ]:
        table = pd.read_feather(processed_dir / table_name)
        result = result.merge(table, on="sample_id", how="left", validate="one_to_one")
        del table
    result["trade_order_pressure_gap_60"] = (
        result["trade_volume_imbalance_60"] - result["net_order_pressure_60"]
    ).astype(np.float32)
    result["trade_order_pressure_product_60"] = (
        result["trade_volume_imbalance_60"] * result["net_order_pressure_60"]
    ).astype(np.float32)
    result["trade_order_pressure_gap_20"] = (
        result["trade_volume_imbalance_20"] - result["net_order_pressure_20"]
    ).astype(np.float32)
    result["trade_order_pressure_product_20"] = (
        result["trade_volume_imbalance_20"] * result["net_order_pressure_20"]
    ).astype(np.float32)
    gc.collect()
    return result


def model_parameters(**updates) -> dict:
    """Return the unchanged EXP-TREE-025 parameters plus one declared update."""

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
    parameters.update(updates)
    return parameters


def subset_cosine(predictions: pd.DataFrame, query: str) -> float:
    subset = predictions.query(query)
    return cosine_similarity_score(subset["target"], subset["prediction"])


def centered_cosine(left, right) -> float:
    """Compute correlation-style cosine after removing both means."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    return float(np.dot(left_values, right_values) / denominator) if denominator else 0.0


def rank_normalized_target_score(predictions: pd.DataFrame, query: str) -> float:
    """Use within-month target ranks as an alternate, non-official diagnostic."""

    subset = predictions.query(query).copy()
    subset["target_rank_normalized"] = (
        subset.groupby("month", sort=False)["target"].rank(pct=True, method="average")
        - 0.5
    ) * 2.0
    return centered_cosine(subset["target_rank_normalized"], subset["prediction"])


def monthly_spearman_macro(predictions: pd.DataFrame, query: str) -> float:
    """Average monthly rank correlation without weighting large months more heavily."""

    subset = predictions.query(query)
    scores = []
    for _, month_data in subset.groupby("month", sort=True):
        score = month_data["target"].corr(month_data["prediction"], method="spearman")
        if np.isfinite(score):
            scores.append(float(score))
    return float(np.mean(scores))


def leave_one_month_out(
    baseline: pd.DataFrame, candidate: pd.DataFrame, query: str | None = None
) -> pd.DataFrame:
    """Recompute whole-vector cosine after excluding each validation month."""

    if query is not None:
        baseline = baseline.query(query)
        candidate = candidate.query(query)
    rows = []
    for month in sorted(candidate["month"].unique()):
        baseline_subset = baseline[baseline["month"] != month]
        candidate_subset = candidate[candidate["month"] != month]
        baseline_score = cosine_similarity_score(
            baseline_subset["target"], baseline_subset["prediction"]
        )
        candidate_score = cosine_similarity_score(
            candidate_subset["target"], candidate_subset["prediction"]
        )
        rows.append(
            {
                "excluded_month": int(month),
                "baseline_cosine": baseline_score,
                "candidate_cosine": candidate_score,
                "change": candidate_score - baseline_score,
            }
        )
    return pd.DataFrame(rows)


def evaluate_and_save(
    *,
    project_dir: Path,
    experiment_id: str,
    description: str,
    model: XGBRegressor,
    predictions: np.ndarray,
    validation_rows: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    training_seconds: float,
    baseline_id: str = BASELINE_ID,
    prediction_iteration: int | None = None,
    save_model: bool = True,
) -> dict:
    prediction_data = validation_rows.copy()
    prediction_data["prediction"] = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(prediction_data["prediction"]).all():
        raise AssertionError(f"Non-finite predictions in {experiment_id}.")
    assert_prediction_alignment(baseline_predictions, prediction_data)
    monthly_scores = make_monthly_scores(prediction_data)
    monthly_comparison = compare_months(baseline_predictions, monthly_scores)
    leave_month_out = leave_one_month_out(baseline_predictions, prediction_data)
    primary_leave_month_out = leave_one_month_out(
        baseline_predictions,
        prediction_data,
        query="month >= 62 and month != 66",
    )

    run_dir = project_dir / "data" / "interim" / "tree_experiments" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = project_dir / "outputs" / "predictions" / f"{experiment_id.lower()}_valid.feather"
    model_path = project_dir / "outputs" / "models" / f"{experiment_id.lower()}.json"
    prediction_data.to_feather(prediction_path)
    monthly_scores.to_csv(run_dir / "monthly_cosine.csv", index=False)
    monthly_comparison.to_csv(run_dir / "monthly_comparison.csv", index=False)
    leave_month_out.to_csv(run_dir / "leave_one_month_out.csv", index=False)
    primary_leave_month_out.to_csv(
        run_dir / "primary_leave_one_month_out.csv", index=False
    )
    if save_model:
        save_model_safely(model, model_path)

    overall = cosine_similarity_score(prediction_data["target"], prediction_data["prediction"])
    stability = summarize_monthly_stability(monthly_scores)
    baseline_monthly_scores = make_monthly_scores(baseline_predictions)
    primary_monthly_scores = monthly_scores.query("month >= 62 and month != 66")
    baseline_primary_monthly_scores = baseline_monthly_scores.query(
        "month >= 62 and month != 66"
    )
    recent_score = subset_cosine(prediction_data, "month >= 67")
    early_score = subset_cosine(prediction_data, "month <= 64")
    without_66_score = subset_cosine(prediction_data, "month >= 62 and month != 66")
    baseline_recent_score = subset_cosine(baseline_predictions, "month >= 67")
    baseline_early_score = subset_cosine(baseline_predictions, "month <= 64")
    baseline_without_66_score = subset_cosine(
        baseline_predictions, "month >= 62 and month != 66"
    )
    monthly_worst = float(monthly_scores["cosine"].min())
    monthly_q25 = float(monthly_scores["cosine"].quantile(0.25))
    primary_monthly_worst = float(primary_monthly_scores["cosine"].min())
    primary_monthly_q25 = float(primary_monthly_scores["cosine"].quantile(0.25))
    primary_monthly_std = float(primary_monthly_scores["cosine"].std(ddof=0))
    baseline_monthly_worst = float(baseline_monthly_scores["cosine"].min())
    baseline_monthly_q25 = float(baseline_monthly_scores["cosine"].quantile(0.25))
    baseline_primary_monthly_worst = float(
        baseline_primary_monthly_scores["cosine"].min()
    )
    baseline_primary_monthly_q25 = float(
        baseline_primary_monthly_scores["cosine"].quantile(0.25)
    )
    passes_core_gates = bool(
        without_66_score > baseline_without_66_score
        and recent_score > baseline_recent_score
        and early_score >= baseline_early_score - 0.001
        and primary_leave_month_out["change"].min() >= 0
    )
    passes_strict_stability_gates = bool(
        passes_core_gates
        and primary_monthly_worst >= baseline_primary_monthly_worst
        and primary_monthly_q25 >= baseline_primary_monthly_q25
    )
    metadata = {
        "experiment_id": experiment_id,
        "description": description,
        "baseline": baseline_id,
        "train_months": "0-59",
        "validation_months": "60-70",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_centered": True,
        "parameters": parameters,
        "prediction_iteration": prediction_iteration,
        "training_seconds": training_seconds,
        "xgboost_version": xgboost.__version__,
        "overall_cosine": overall,
        "cosine_62_70": subset_cosine(prediction_data, "month >= 62"),
        "cosine_62_70_without_66": without_66_score,
        "cosine_67_70": recent_score,
        "cosine_60_64": early_score,
        "monthly_stability": stability,
        "monthly_worst": monthly_worst,
        "monthly_q25": monthly_q25,
        "primary_monthly_worst": primary_monthly_worst,
        "primary_monthly_q25": primary_monthly_q25,
        "primary_monthly_std": primary_monthly_std,
        "improved_month_count": int((monthly_comparison["change"] > 0).sum()),
        "declined_month_count": int((monthly_comparison["change"] < 0).sum()),
        "leave_one_month_out_min_change": float(leave_month_out["change"].min()),
        "leave_one_month_out_max_change": float(leave_month_out["change"].max()),
        "primary_leave_one_month_out_min_change": float(
            primary_leave_month_out["change"].min()
        ),
        "primary_leave_one_month_out_mean_change": float(
            primary_leave_month_out["change"].mean()
        ),
        "rank_normalized_target_cosine_67_70": rank_normalized_target_score(
            prediction_data, "month >= 67"
        ),
        "monthly_spearman_macro_67_70": monthly_spearman_macro(
            prediction_data, "month >= 67"
        ),
        "passes_core_generalization_gates": passes_core_gates,
        "passes_strict_stability_gates": passes_strict_stability_gates,
        "prediction_path": str(prediction_path),
        "model_path": str(model_path) if save_model else None,
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{experiment_id}: overall={overall:.10f}, "
        f"62-70={metadata['cosine_62_70']:.10f}, "
        f"62-70_no66={metadata['cosine_62_70_without_66']:.10f}, "
        f"std={stability['monthly_population_std']:.10f}, "
        f"months_up={metadata['improved_month_count']}/11, "
        f"primary_LOMO_min={metadata['primary_leave_one_month_out_min_change']:+.10f}, "
        f"core_gate={passes_core_gates}, strict_gate={passes_strict_stability_gates}",
        flush=True,
    )
    return metadata


def fit_model(
    model_data: pd.DataFrame,
    feature_columns: list[str],
    train_mask: pd.Series,
    centered_target: pd.Series,
    parameters: dict,
    sample_weight: np.ndarray | None = None,
) -> tuple[XGBRegressor, float]:
    model = XGBRegressor(**parameters)
    start_time = time.perf_counter()
    model.fit(
        model_data.loc[train_mask, feature_columns],
        centered_target,
        sample_weight=sample_weight,
    )
    return model, time.perf_counter() - start_time


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)
    labels = pd.read_feather(project_dir / "data" / "raw" / "label.feather")
    model_data = model_data.merge(labels, on="sample_id", how="inner", validate="one_to_one")
    del labels
    gc.collect()

    exp025_features = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
    )
    if len(exp025_features) != 163 or len(exp025_features) != len(set(exp025_features)):
        raise AssertionError("Expected the exact 163-feature EXP-TREE-025 schema.")
    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    validation_rows = model_data.loc[valid_mask, ["sample_id", "month", "target"]].copy()
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "xgboost_order_multiwindow_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    results = []

    # 一次训练800棵树，再读取预先约定的200/400/800阶段。
    # Train 800 trees once, then inspect the predetermined 200/400/800 prefixes.
    rounds_parameters = model_parameters(n_estimators=800)
    rounds_model, rounds_seconds = fit_model(
        model_data, exp025_features, train_mask, centered_target, rounds_parameters
    )
    prediction_200 = rounds_model.predict(
        model_data.loc[valid_mask, exp025_features], iteration_range=(0, 200)
    )
    prediction_400 = rounds_model.predict(
        model_data.loc[valid_mask, exp025_features], iteration_range=(0, 400)
    )
    prediction_800 = rounds_model.predict(model_data.loc[valid_mask, exp025_features])
    baseline_difference = float(
        np.max(np.abs(prediction_400 - baseline_predictions["prediction"].to_numpy()))
    )
    if baseline_difference > 1e-12:
        raise AssertionError(f"The 400-tree prefix did not reproduce EXP-TREE-025: {baseline_difference}")
    results.append(
        evaluate_and_save(
            project_dir=project_dir,
            experiment_id="EXP-TREE-027",
            description="EXP025 tree-count curve; evaluate the 200-tree prefix of one 800-tree fit",
            model=rounds_model,
            predictions=prediction_200,
            validation_rows=validation_rows,
            baseline_predictions=baseline_predictions,
            feature_columns=exp025_features,
            parameters=rounds_parameters,
            training_seconds=rounds_seconds,
            prediction_iteration=200,
            save_model=False,
        )
    )
    results.append(
        evaluate_and_save(
            project_dir=project_dir,
            experiment_id="EXP-TREE-028",
            description="EXP025 tree-count curve; evaluate all 800 trees",
            model=rounds_model,
            predictions=prediction_800,
            validation_rows=validation_rows,
            baseline_predictions=baseline_predictions,
            feature_columns=exp025_features,
            parameters=rounds_parameters,
            training_seconds=rounds_seconds,
            prediction_iteration=800,
            save_model=True,
        )
    )
    del prediction_200, prediction_400, prediction_800, rounds_model
    gc.collect()

    variants = [
        (
            "EXP-TREE-029",
            "change only colsample_bytree from 1.0 to 0.8",
            exp025_features,
            model_parameters(colsample_bytree=0.8),
        ),
        (
            "EXP-TREE-030",
            "remove the legacy eight order-flow columns and retain order multi-window columns",
            [column for column in exp025_features if column not in ORDER_FEATURE_COLUMNS],
            model_parameters(),
        ),
        (
            "EXP-TREE-031",
            "change only min_child_weight from 100 to 500",
            exp025_features,
            model_parameters(min_child_weight=500.0),
        ),
        (
            "EXP-TREE-032",
            "add four transaction-pressure by order-pressure gap/product features",
            exp025_features + CROSS_FEATURE_COLUMNS,
            model_parameters(),
        ),
    ]
    for experiment_id, description, feature_columns, parameters in variants:
        if len(feature_columns) != len(set(feature_columns)):
            raise AssertionError(f"Duplicate feature columns in {experiment_id}.")
        model, training_seconds = fit_model(
            model_data, feature_columns, train_mask, centered_target, parameters
        )
        predictions = model.predict(model_data.loc[valid_mask, feature_columns])
        results.append(
            evaluate_and_save(
                project_dir=project_dir,
                experiment_id=experiment_id,
                description=description,
                model=model,
                predictions=predictions,
                validation_rows=validation_rows,
                baseline_predictions=baseline_predictions,
                feature_columns=feature_columns,
                parameters=parameters,
                training_seconds=training_seconds,
            )
        )
        del model, predictions
        gc.collect()

    summary = pd.DataFrame(
        [
            {
                "experiment_id": item["experiment_id"],
                "description": item["description"],
                "feature_count": item["feature_count"],
                "overall_cosine": item["overall_cosine"],
                "cosine_62_70": item["cosine_62_70"],
                "cosine_62_70_without_66": item["cosine_62_70_without_66"],
                "monthly_std": item["monthly_stability"]["monthly_population_std"],
                "worst_month": item["monthly_stability"]["worst_month"],
                "worst_month_cosine": item["monthly_stability"]["worst_month_cosine"],
                "improved_month_count": item["improved_month_count"],
                "leave_one_month_out_min_change": item["leave_one_month_out_min_change"],
            }
            for item in results
        ]
    )
    summary_path = project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-027-032-summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
