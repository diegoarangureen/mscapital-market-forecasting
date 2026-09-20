"""Evaluate saved tree predictions with multiple temporal generalization diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
    cosine_similarity_score,
    make_monthly_scores,
)


EXPERIMENT_IDS = [
    "EXP-TREE-034",
    "EXP-TREE-036",
    "EXP-TREE-037",
    "EXP-TREE-038",
    "EXP-TREE-039",
    "EXP-TREE-040",
    "EXP-TREE-041",
    "EXP-TREE-042",
    "EXP-TREE-043",
    "EXP-TREE-044",
    "EXP-TREE-045",
    "EXP-TREE-046",
    "EXP-TREE-047",
    "EXP-TREE-048",
    "EXP-TREE-049",
    "EXP-TREE-050",
    "EXP-TREE-051",
]
BASELINE_ID = "EXP-TREE-037"

# 允许候选模型在较早月份有极小回撤，但不能用明显牺牲旧阶段来换近期分数。
# Allow a tiny early-period regression, but reject material sacrifice of older months.
MAX_EARLY_REGRESSION = 0.001


def centered_cosine(left, right) -> float:
    # Copy because Series.to_numpy() may expose a writable view of the DataFrame.
    # 复制数组，避免减均值时意外修改原始预测列。
    left_values = np.asarray(left, dtype=np.float64).copy()
    right_values = np.asarray(right, dtype=np.float64).copy()
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    return float(np.dot(left_values, right_values) / denominator) if denominator else 0.0


def score_query(predictions: pd.DataFrame, query: str) -> float:
    subset = predictions.query(query)
    return cosine_similarity_score(subset["target"], subset["prediction"])


def rank_normalized_target_score(predictions: pd.DataFrame, query: str) -> float:
    subset = predictions.query(query).copy()
    subset["target_rank_normalized"] = (
        subset.groupby("month", sort=False)["target"].rank(pct=True, method="average")
        - 0.5
    ) * 2.0
    return centered_cosine(subset["target_rank_normalized"], subset["prediction"])


def monthly_spearman_macro(predictions: pd.DataFrame, query: str) -> float:
    subset = predictions.query(query)
    scores = []
    for _, month_data in subset.groupby("month", sort=True):
        score = month_data["target"].corr(month_data["prediction"], method="spearman")
        if np.isfinite(score):
            scores.append(float(score))
    return float(np.mean(scores))


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    baseline = pd.read_feather(
        prediction_dir / f"{BASELINE_ID.lower()}_valid.feather"
    )
    baseline_monthly = make_monthly_scores(baseline).set_index("month")["cosine"]
    baseline_primary_monthly = baseline_monthly[
        (baseline_monthly.index >= 62) & (baseline_monthly.index != 66)
    ]
    baseline_std = float(baseline_monthly.std(ddof=0))
    baseline_prediction_std = float(baseline["prediction"].std(ddof=0))
    rows = []

    for experiment_id in EXPERIMENT_IDS:
        predictions = pd.read_feather(
            prediction_dir / f"{experiment_id.lower()}_valid.feather"
        )
        assert_prediction_alignment(baseline, predictions)
        monthly = make_monthly_scores(predictions).set_index("month")["cosine"]
        leave_one_month_changes = []
        for month in sorted(predictions["month"].unique()):
            baseline_without = baseline[baseline["month"] != month]
            candidate_without = predictions[predictions["month"] != month]
            leave_one_month_changes.append(
                cosine_similarity_score(
                    candidate_without["target"], candidate_without["prediction"]
                )
                - cosine_similarity_score(
                    baseline_without["target"], baseline_without["prediction"]
                )
            )
        primary_baseline = baseline.query("month >= 62 and month != 66")
        primary_candidate = predictions.query("month >= 62 and month != 66")
        primary_leave_one_month_changes = []
        for month in sorted(primary_candidate["month"].unique()):
            baseline_without = primary_baseline[primary_baseline["month"] != month]
            candidate_without = primary_candidate[primary_candidate["month"] != month]
            primary_leave_one_month_changes.append(
                cosine_similarity_score(
                    candidate_without["target"], candidate_without["prediction"]
                )
                - cosine_similarity_score(
                    baseline_without["target"], baseline_without["prediction"]
                )
            )
        primary_monthly = monthly[(monthly.index >= 62) & (monthly.index != 66)]
        rows.append(
            {
                "experiment_id": experiment_id,
                "official_60_70": score_query(predictions, "month >= 60"),
                "official_62_70_without_66": score_query(
                    predictions, "month >= 62 and month != 66"
                ),
                "official_recent_67_70": score_query(predictions, "month >= 67"),
                "official_early_60_64": score_query(predictions, "month <= 64"),
                "monthly_macro_mean": float(monthly.mean()),
                "monthly_std": float(monthly.std(ddof=0)),
                "monthly_std_change": float(monthly.std(ddof=0)) - baseline_std,
                "monthly_worst": float(monthly.min()),
                "monthly_q25": float(monthly.quantile(0.25)),
                "primary_monthly_worst": float(primary_monthly.min()),
                "primary_monthly_q25": float(primary_monthly.quantile(0.25)),
                "primary_monthly_std": float(primary_monthly.std(ddof=0)),
                "months_better_than_baseline": int((monthly > baseline_monthly).sum()),
                "leave_one_month_out_min_change": float(np.min(leave_one_month_changes)),
                "leave_one_month_out_mean_change": float(np.mean(leave_one_month_changes)),
                "primary_leave_one_month_out_min_change": float(
                    np.min(primary_leave_one_month_changes)
                ),
                "primary_leave_one_month_out_mean_change": float(
                    np.mean(primary_leave_one_month_changes)
                ),
                "rank_normalized_target_cosine_67_70": rank_normalized_target_score(
                    predictions, "month >= 67"
                ),
                "monthly_spearman_macro_67_70": monthly_spearman_macro(
                    predictions, "month >= 67"
                ),
                "prediction_correlation_with_baseline": float(
                    centered_cosine(
                        predictions["prediction"].to_numpy(),
                        baseline["prediction"].to_numpy(),
                    )
                ),
                "prediction_std_ratio_to_baseline": float(
                    predictions["prediction"].std(ddof=0) / baseline_prediction_std
                ),
            }
        )

    panel = pd.DataFrame(rows)
    baseline_row = panel.loc[panel["experiment_id"] == BASELINE_ID].iloc[0]

    # 基本闸门关注不同时间窗口是否同时成立，而不是只追一个汇总 cosine。
    # Core gates require gains across different time windows, not one aggregate cosine.
    panel["gate_primary_without_66"] = (
        panel["official_62_70_without_66"]
        > baseline_row["official_62_70_without_66"]
    )
    panel["gate_recent_67_70"] = (
        panel["official_recent_67_70"] > baseline_row["official_recent_67_70"]
    )
    panel["gate_early_60_64"] = (
        panel["official_early_60_64"]
        >= baseline_row["official_early_60_64"] - MAX_EARLY_REGRESSION
    )
    panel["gate_leave_one_month_out"] = (
        panel["primary_leave_one_month_out_min_change"] >= 0
    )
    panel["passes_core_gates"] = panel[
        [
            "gate_primary_without_66",
            "gate_recent_67_70",
            "gate_early_60_64",
            "gate_leave_one_month_out",
        ]
    ].all(axis=1)

    # 严格闸门再要求分布较差的月份不退步。标准差只作诊断，因为“稳定地差”也会很低。
    # Strict gates also protect weak months. Std stays diagnostic: uniformly poor can look stable.
    panel["gate_monthly_worst"] = (
        panel["primary_monthly_worst"] >= baseline_primary_monthly.min()
    )
    panel["gate_monthly_q25"] = (
        panel["primary_monthly_q25"] >= baseline_primary_monthly.quantile(0.25)
    )
    panel["passes_strict_stability_gates"] = panel[
        ["passes_core_gates", "gate_monthly_worst", "gate_monthly_q25"]
    ].all(axis=1)
    panel.loc[panel["experiment_id"] == BASELINE_ID, "passes_core_gates"] = False
    panel.loc[
        panel["experiment_id"] == BASELINE_ID, "passes_strict_stability_gates"
    ] = False

    panel = panel.sort_values(
        ["official_62_70_without_66", "leave_one_month_out_min_change"],
        ascending=False,
    )
    output_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "generalization_panel_exp037_051.json"
    )
    output_path.write_text(
        json.dumps(panel.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(panel.to_string(index=False), flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
