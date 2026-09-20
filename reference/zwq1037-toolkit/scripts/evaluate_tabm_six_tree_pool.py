"""Coarsely screen six tree families around the robust dual-loss TabM blend."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-BLEND-004"
TABM_WEIGHTS = {
    "EXP-TABM-001-MSE": 0.55,
    "EXP-TABM-003-COSINE": 0.15,
}
TREE_WEIGHT_TOTAL = 0.30
TREE_WEIGHT_STEP = 0.05
SOURCE_FILES = {
    "EXP-TABM-001-MSE": "exp-tabm-001_valid.feather",
    "EXP-TABM-003-COSINE": "exp-tabm-003-cosine_valid.feather",
    "EXP-TREE-053R-XGBOOST": "exp-tree-053r_valid.feather",
    "EXP-TREE-068-LIGHTGBM": "exp-tree-068_valid.feather",
    "EXP-TREE-007-HISTGB": "hist_gradient_boosting_target_centered_level2_valid.feather",
    "EXP-TREE-069-CATBOOST": "exp-tree-069_valid.feather",
    "EXTRA-TREES-LEVEL2": "extra_trees_target_centered_level2_valid.feather",
    "RANDOM-FOREST": "random_forest_target_centered_valid.feather",
}
TREE_NAMES = list(SOURCE_FILES)[2:]
NEW_TREE_NAMES = TREE_NAMES[3:]
BASELINE_WEIGHTS = {
    **TABM_WEIGHTS,
    "EXP-TREE-053R-XGBOOST": 0.10,
    "EXP-TREE-068-LIGHTGBM": 0.10,
    "EXP-TREE-007-HISTGB": 0.10,
    "EXP-TREE-069-CATBOOST": 0.00,
    "EXTRA-TREES-LEVEL2": 0.00,
    "RANDOM-FOREST": 0.00,
}
SUBMITTED_OVERALL_FLOOR = 0.17571880874787868


def integer_compositions(total: int, parts: int):
    """Yield non-negative integer vectors whose sum is ``total``."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in integer_compositions(total - first, parts - 1):
            yield (first, *rest)


def precompute_score_terms(
    target: np.ndarray, normalized_predictions: np.ndarray, mask: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Precompute cosine numerator and denominator terms for one row subset."""
    selected_target = target[mask]
    selected_predictions = normalized_predictions[mask]
    target_norm = float(np.linalg.norm(selected_target))
    target_dots = selected_predictions.T @ selected_target
    prediction_gram = selected_predictions.T @ selected_predictions
    return target_norm, target_dots, prediction_gram


def score_from_terms(
    weights: np.ndarray, terms: tuple[float, np.ndarray, np.ndarray]
) -> float:
    """Calculate a subset cosine score from cached matrix products."""
    target_norm, target_dots, prediction_gram = terms
    prediction_norm_squared = float(weights @ prediction_gram @ weights)
    denominator = target_norm * np.sqrt(max(prediction_norm_squared, 0.0))
    return float(weights @ target_dots / denominator) if denominator else 0.0


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    model_names = list(SOURCE_FILES)
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["EXP-TABM-001-MSE"]
    validation_rows = reference[["sample_id", "month", "target"]].copy()
    for frame in frames.values():
        assert_prediction_alignment(frame, validation_rows)

    # 每个来源先做全局 L2 归一化，避免预测尺度替代真正的融合权重。
    # Normalize every source globally so prediction scale cannot replace blend weight.
    normalized_columns = []
    source_norms = {}
    for name in model_names:
        prediction = frames[name]["prediction"].to_numpy(dtype=np.float64)
        source_norms[name] = float(np.linalg.norm(prediction))
        normalized_columns.append(prediction / source_norms[name])
    normalized_predictions = np.column_stack(normalized_columns)
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    month_values = np.sort(np.unique(months))

    masks = {
        "overall_cosine": np.ones(len(months), dtype=bool),
        "cosine_62_70_without_66": (months >= 62) & (months != 66),
        "cosine_67_70": months >= 67,
        "cosine_60_64": months <= 64,
    }
    for month in month_values:
        masks[f"month_{int(month)}"] = months == month
    primary_months = [int(month) for month in month_values if month >= 62 and month != 66]
    for omitted_month in primary_months:
        masks[f"primary_without_{omitted_month}"] = (
            (months >= 62) & (months != 66) & (months != omitted_month)
        )
    terms = {
        name: precompute_score_terms(target, normalized_predictions, mask)
        for name, mask in masks.items()
    }

    baseline_vector = np.asarray(
        [BASELINE_WEIGHTS.get(name, 0.0) for name in model_names], dtype=np.float64
    )
    baseline_metrics = {
        name: score_from_terms(baseline_vector, terms[name])
        for name in masks
    }

    units = int(round(TREE_WEIGHT_TOTAL / TREE_WEIGHT_STEP))
    rows = []
    for composition in integer_compositions(units, len(TREE_NAMES)):
        weights = dict(TABM_WEIGHTS)
        weights.update(
            {
                name: count * TREE_WEIGHT_STEP
                for name, count in zip(TREE_NAMES, composition, strict=True)
            }
        )
        weight_vector = np.asarray([weights[name] for name in model_names])
        row = {f"weight_{name}": weights[name] for name in model_names}
        for metric_name in [
            "overall_cosine",
            "cosine_62_70_without_66",
            "cosine_67_70",
            "cosine_60_64",
        ]:
            row[metric_name] = score_from_terms(weight_vector, terms[metric_name])
        monthly_scores = np.asarray(
            [
                score_from_terms(weight_vector, terms[f"month_{int(month)}"])
                for month in month_values
            ]
        )
        primary_scores = monthly_scores[
            np.asarray([(month >= 62 and month != 66) for month in month_values])
        ]
        row["primary_monthly_std"] = float(primary_scores.std(ddof=0))
        row["primary_monthly_worst"] = float(primary_scores.min())
        row["primary_monthly_q25"] = float(np.quantile(primary_scores, 0.25))
        lomo_changes = [
            score_from_terms(weight_vector, terms[f"primary_without_{month}"])
            - baseline_metrics[f"primary_without_{month}"]
            for month in primary_months
        ]
        row["primary_lomo_min_change_vs_blend003"] = float(np.min(lomo_changes))
        row["primary_lomo_mean_change_vs_blend003"] = float(np.mean(lomo_changes))
        row["uses_new_tree_family"] = any(weights[name] > 0 for name in NEW_TREE_NAMES)
        rows.append(row)

    grid = pd.DataFrame(rows)
    baseline_summary = {
        "overall_cosine": baseline_metrics["overall_cosine"],
        "cosine_62_70_without_66": baseline_metrics["cosine_62_70_without_66"],
        "cosine_67_70": baseline_metrics["cosine_67_70"],
        "cosine_60_64": baseline_metrics["cosine_60_64"],
    }
    baseline_monthly = np.asarray(
        [baseline_metrics[f"month_{int(month)}"] for month in month_values]
    )
    baseline_primary = baseline_monthly[
        np.asarray([(month >= 62 and month != 66) for month in month_values])
    ]
    baseline_summary.update(
        {
            "primary_monthly_std": float(baseline_primary.std(ddof=0)),
            "primary_monthly_worst": float(baseline_primary.min()),
            "primary_monthly_q25": float(np.quantile(baseline_primary, 0.25)),
        }
    )

    # 只接受多时间窗口同时成立的方案；允许旧窗口最多回撤 0.0005。
    # Accept only multi-window gains; tolerate at most 0.0005 on the older window.
    eligible = grid[
        grid["uses_new_tree_family"]
        & (grid["overall_cosine"] >= SUBMITTED_OVERALL_FLOOR)
        & (
            grid["cosine_62_70_without_66"]
            >= baseline_summary["cosine_62_70_without_66"]
        )
        & (grid["cosine_67_70"] >= baseline_summary["cosine_67_70"])
        & (grid["cosine_60_64"] >= baseline_summary["cosine_60_64"] - 0.0005)
        & (
            grid["primary_monthly_worst"]
            >= baseline_summary["primary_monthly_worst"]
        )
        & (
            grid["primary_monthly_q25"]
            >= baseline_summary["primary_monthly_q25"] - 0.0001
        )
        & (grid["primary_lomo_min_change_vs_blend003"] >= 0.0)
    ].copy()
    if not eligible.empty:
        eligible["minimum_key_window_gain"] = np.minimum(
            eligible["cosine_62_70_without_66"]
            - baseline_summary["cosine_62_70_without_66"],
            eligible["cosine_67_70"] - baseline_summary["cosine_67_70"],
        )
        selected = eligible.sort_values(
            [
                "minimum_key_window_gain",
                "primary_monthly_q25",
                "overall_cosine",
            ],
            ascending=False,
        ).iloc[0]
        selection_status = "selected_extended_tree_pool_candidate"
    else:
        baseline_mask = np.ones(len(grid), dtype=bool)
        for name, weight in BASELINE_WEIGHTS.items():
            baseline_mask &= np.isclose(grid[f"weight_{name}"], weight)
        selected = grid.loc[baseline_mask].iloc[0]
        selection_status = "no_extended_tree_pool_candidate_passed_all_gates"

    selected_weights = {
        name: float(selected[f"weight_{name}"]) for name in model_names
    }
    selected_vector = np.asarray([selected_weights[name] for name in model_names])
    selected_prediction = normalized_predictions @ selected_vector

    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = grid.sort_values(
        ["cosine_62_70_without_66", "cosine_67_70", "overall_cosine"],
        ascending=False,
    )
    grid.to_csv(run_dir / "blend_grid.csv", index=False)
    eligible.to_csv(run_dir / "eligible_candidates.csv", index=False)
    correlation = pd.DataFrame(
        np.corrcoef(normalized_predictions, rowvar=False),
        index=model_names,
        columns=model_names,
    )
    correlation.to_csv(run_dir / "source_prediction_correlations.csv")

    baseline_frame = pd.read_feather(prediction_dir / "exp-blend-003_valid.feather")
    assert_prediction_alignment(baseline_frame, validation_rows)
    tabm_config = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-001"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "fixed 55% MSE TabM plus 15% cosine TabM; coarse 0.05 grid allocates "
            "30% across XGBoost, LightGBM, HistGradientBoosting, CatBoost, "
            "ExtraTrees, and RandomForest"
        ),
        model=None,
        predictions=selected_prediction,
        validation_rows=validation_rows,
        baseline_predictions=baseline_frame,
        feature_columns=list(tabm_config["feature_columns"]),
        parameters={
            "selected_weights": selected_weights,
            "tree_weight_total": TREE_WEIGHT_TOTAL,
            "tree_weight_step": TREE_WEIGHT_STEP,
            "selection_status": selection_status,
            "selection_rule": (
                "new tree required; overall above submitted blend001; no66/recent/"
                "worst and primary LOMO non-decreasing versus blend003; early "
                "tolerance 0.0005; q25 tolerance 0.0001; maximize minimum no66/"
                "recent gain"
            ),
        },
        training_seconds=0.0,
        baseline_id="EXP-BLEND-003",
        save_model=False,
    )
    metadata.update(
        {
            "model_family": "normalized_prediction_blend",
            "source_models": model_names,
            "source_prediction_files": SOURCE_FILES,
            "source_prediction_norms": source_norms,
            "baseline_summary": baseline_summary,
            "grid_candidate_count": int(len(grid)),
            "eligible_candidate_count": int(len(eligible)),
            "blend_grid_path": str(run_dir / "blend_grid.csv"),
            "eligible_candidates_path": str(run_dir / "eligible_candidates.csv"),
            "correlation_path": str(run_dir / "source_prediction_correlations.csv"),
        }
    )
    metadata.pop("xgboost_version", None)
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"selection_status={selection_status}", flush=True)
    print(f"selected_weights={selected_weights}", flush=True)
    print(f"eligible_candidates={len(eligible)} / {len(grid)}", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
