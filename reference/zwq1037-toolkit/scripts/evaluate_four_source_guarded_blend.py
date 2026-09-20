"""Save the four-source blend selected by every-primary-month guardrails."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-BLEND-010"
SOURCE_FILES = {
    "tabm_original_centered": "exp-tabm-001_valid.feather",
    "tabm_corrprune_raw": "exp-tabm-006-corrprune_valid.feather",
    "tabm_cosine_raw": "exp-tabm-003-cosine_valid.feather",
    "xgboost_centered": "exp-tree-053r_valid.feather",
}
WEIGHTS = {
    "tabm_original_centered": 0.50,
    "tabm_corrprune_raw": 0.25,
    "tabm_cosine_raw": 0.05,
    "xgboost_centered": 0.20,
}
BASELINE_WEIGHTS = {
    "tabm_original_centered": 0.80,
    "tabm_corrprune_raw": 0.00,
    "tabm_cosine_raw": 0.00,
    "xgboost_centered": 0.20,
}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_original_centered"]
    validation_rows = reference[["sample_id", "month", "target"]].copy()
    normalized = {}
    source_norms = {}
    source_means_removed = {}
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        prediction = frame["prediction"].to_numpy(dtype=np.float64)
        if name.endswith("_centered"):
            source_means_removed[name] = float(prediction.mean())
            prediction = prediction - prediction.mean()
        else:
            source_means_removed[name] = 0.0
        source_norms[name] = float(np.linalg.norm(prediction))
        normalized[name] = prediction / source_norms[name]
    selected_prediction = sum(WEIGHTS[name] * normalized[name] for name in WEIGHTS)
    baseline_prediction = sum(
        BASELINE_WEIGHTS[name] * normalized[name] for name in BASELINE_WEIGHTS
    )
    baseline_frame = validation_rows.copy()
    baseline_frame["prediction"] = baseline_prediction

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
            "50% centered original TabM, 25% raw correlation-pruned TabM, "
            "5% raw cosine-loss TabM, and 20% centered XGBoost; selected with "
            "a non-regression guard for every primary month"
        ),
        model=None,
        predictions=selected_prediction,
        validation_rows=validation_rows,
        baseline_predictions=baseline_frame,
        feature_columns=list(tabm_config["feature_columns"]),
        parameters={
            "weights": WEIGHTS,
            "baseline_weights": BASELINE_WEIGHTS,
            "normalization": "per-source L2 after selective label-free centering",
            "selection_rule": (
                "0.05 grid; overall/no66/recent non-decreasing, early tolerance "
                "0.0002, and every month 62-70 except 66 non-decreasing versus "
                "centered 80/20"
            ),
        },
        training_seconds=0.0,
        baseline_id="EXP-POST-002-CENTERED80",
        save_model=False,
    )
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    metadata.update(
        {
            "model_family": "selectively_centered_normalized_prediction_blend",
            "source_prediction_files": SOURCE_FILES,
            "source_prediction_norms": source_norms,
            "source_means_removed": source_means_removed,
            "monthwise_guard_result": str(
                project_dir
                / "data"
                / "interim"
                / "tree_experiments"
                / "EXP-BLEND-009-MONTHWISE-GUARD"
                / "result.json"
            ),
        }
    )
    metadata.pop("xgboost_version", None)
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
