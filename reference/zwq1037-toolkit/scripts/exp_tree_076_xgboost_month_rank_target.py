"""Test within-month target ranks with the same Relative319 XGBoost setup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_027_031_xgboost_capacity_regularization import fit_model, model_parameters


EXPERIMENT_ID = "EXP-TREE-076-RELATIVE319-MONTH-RANK-TARGET"
FOLDS = {
    "dev": ("train049_valid5059", 49, 50, 59),
    "confirm": ("train059_valid6270_no66", 59, 62, 70),
}


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=FOLDS, required=True)
    stage = parser.parse_args().stage
    fold, train_end, valid_start, valid_end = FOLDS[stage]
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID / fold
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, baseline_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, frame)
    feature_columns = [*baseline_columns, *RELATIVE_COLUMNS]
    months = frame["month"].to_numpy(dtype=np.int16)
    train_mask = months <= train_end
    valid_mask = (months >= valid_start) & (months <= valid_end)
    if stage == "confirm":
        valid_mask &= months != 66
    raw_target = frame["target"].to_numpy(dtype=np.float64)
    ranked_target = np.empty(int(train_mask.sum()), dtype=np.float32)
    # 仅训练月份的答案参与每月排名。 / Rank training targets within each training month only.
    training_months = months[train_mask]
    for month in np.unique(training_months):
        rows = np.flatnonzero(training_months == month)
        ranked_target[rows] = (
            2.0 * rankdata(raw_target[train_mask][rows], method="average")
            / (len(rows) + 1.0) - 1.0
        ).astype(np.float32)
    parameters = model_parameters(
        n_estimators=800, colsample_bytree=0.8, device="cuda", n_jobs=2
    )
    model, training_seconds = fit_model(
        frame, feature_columns, train_mask, pd.Series(ranked_target), parameters
    )
    ids = frame.loc[valid_mask, "sample_id"].to_numpy()
    target = raw_target[valid_mask]
    candidate = np.asarray(
        model.predict(frame.loc[valid_mask, feature_columns]), dtype=np.float64
    )
    baseline_dir = (
        project_dir / "data" / "interim" / "tree_experiments"
        / "EXP-TREE-069-RELATIVE319"
        / ("train049_valid5059" if stage == "dev" else "train059_valid6070")
    )
    baseline_frame = pd.read_feather(
        baseline_dir / "validation_predictions.feather",
        columns=["sample_id", "month", "xgb_relative319"],
    )
    if stage == "confirm":
        baseline_frame = baseline_frame.loc[
            (baseline_frame["month"] >= 62) & (baseline_frame["month"] != 66)
        ]
    if not np.array_equal(ids, baseline_frame["sample_id"].to_numpy()):
        raise AssertionError("Baseline validation IDs differ from candidate IDs.")
    baseline = baseline_frame["xgb_relative319"].to_numpy(dtype=np.float64)
    scores = {"baseline_relative319": cosine(target, baseline),
              "candidate_month_rank_target": cosine(target, candidate)}
    scores["delta"] = scores["candidate_month_rank_target"] - scores["baseline_relative319"]
    result = {
        "experiment_id": EXPERIMENT_ID, "stage": stage,
        "train_months": f"0-{train_end}",
        "purged_months": "60-61" if stage == "confirm" else None,
        "validation_months": f"{valid_start}-{valid_end}",
        "excluded_validation_months": [66] if stage == "confirm" else [],
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "feature_count": len(feature_columns),
        "target_transformation": "within-month percentile rank [-1,1] for training labels",
        "parameters": parameters, "training_seconds": training_seconds,
        "scores": scores,
    }
    pd.DataFrame({
        "sample_id": ids, "month": months[valid_mask], "target": target,
        "baseline_relative319": baseline, "candidate_month_rank_target": candidate,
    }).to_feather(output_dir / "validation_predictions.feather")
    save_model_safely(model, output_dir / "model.json")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": stage, "scores": scores}, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
