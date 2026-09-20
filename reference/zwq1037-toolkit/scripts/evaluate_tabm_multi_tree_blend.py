"""Evaluate a coarse, robustness-first TabM + multi-tree blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-BLEND-002"
SOURCE_FILES = {
    "EXP-TABM-001": "exp-tabm-001_valid.feather",
    "EXP-TREE-053R": "exp-tree-053r_valid.feather",
    "EXP-TREE-068": "exp-tree-068_valid.feather",
    "EXP-TREE-007": "hist_gradient_boosting_target_centered_level2_valid.feather",
}
SELECTED_WEIGHTS = {
    "EXP-TABM-001": 0.70,
    "EXP-TREE-053R": 0.10,
    "EXP-TREE-068": 0.10,
    "EXP-TREE-007": 0.10,
}
TABM_WEIGHT_GRID = [0.70, 0.75, 0.80]


def cosine_score(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction))
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["EXP-TABM-001"]
    validation_rows = reference[["sample_id", "month", "target"]].copy()
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)

    normalized = {}
    source_norms = {}
    for name, frame in frames.items():
        values = frame["prediction"].to_numpy(dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"Invalid prediction norm for {name}.")
        normalized[name] = values / norm
        source_norms[name] = norm

    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    month_values = np.sort(np.unique(months))
    rows = []
    for tabm_weight in TABM_WEIGHT_GRID:
        each_tree_weight = (1.0 - tabm_weight) / 3.0
        weights = {
            "EXP-TABM-001": tabm_weight,
            "EXP-TREE-053R": each_tree_weight,
            "EXP-TREE-068": each_tree_weight,
            "EXP-TREE-007": each_tree_weight,
        }
        prediction = sum(weights[name] * normalized[name] for name in weights)
        monthly_scores = np.asarray(
            [
                cosine_score(target[months == month], prediction[months == month])
                for month in month_values
            ]
        )
        primary = monthly_scores[(month_values >= 62) & (month_values != 66)]
        rows.append(
            {
                "tabm_weight": tabm_weight,
                "each_tree_weight": each_tree_weight,
                "overall_cosine": cosine_score(target, prediction),
                "cosine_62_70_without_66": cosine_score(
                    target[(months >= 62) & (months != 66)],
                    prediction[(months >= 62) & (months != 66)],
                ),
                "cosine_67_70": cosine_score(
                    target[months >= 67], prediction[months >= 67]
                ),
                "cosine_60_64": cosine_score(
                    target[months <= 64], prediction[months <= 64]
                ),
                "primary_monthly_std": float(primary.std(ddof=0)),
                "primary_monthly_worst": float(primary.min()),
                "primary_monthly_q25": float(np.quantile(primary, 0.25)),
            }
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(run_dir / "blend_grid.csv", index=False)

    selected_prediction = sum(
        SELECTED_WEIGHTS[name] * normalized[name] for name in SELECTED_WEIGHTS
    )
    baseline = pd.read_feather(prediction_dir / "exp-blend-001_valid.feather")
    assert_prediction_alignment(baseline, validation_rows)
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
    feature_columns = list(tabm_config["feature_columns"])
    parameters = {
        "weights": SELECTED_WEIGHTS,
        "normalization": "each source validation vector divided by its global L2 norm",
        "selection": (
            "robustness-first choice among a fixed 70/75/80 percent TabM grid; "
            "the remaining weight is equal across three tree families"
        ),
    }
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "70% TabM plus 10% each EXP053R XGBoost, EXP068 LightGBM, and "
            "EXP007 HistGradientBoosting after per-model L2 normalization"
        ),
        model=None,
        predictions=selected_prediction,
        validation_rows=validation_rows,
        baseline_predictions=baseline,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=0.0,
        baseline_id="EXP-BLEND-001",
        save_model=False,
    )
    metadata.update(
        {
            "model_family": "normalized_prediction_blend",
            "training_device": "not_applicable",
            "source_models": list(SOURCE_FILES),
            "source_prediction_files": SOURCE_FILES,
            "source_prediction_norms": source_norms,
            "blend_grid_path": str(run_dir / "blend_grid.csv"),
            "selection_priority": [
                "cosine_62_70_without_66",
                "cosine_67_70",
                "primary_monthly_worst",
                "primary_monthly_q25",
                "primary_monthly_std",
                "overall_cosine",
            ],
        }
    )
    metadata.pop("xgboost_version", None)
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(grid.to_string(index=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
