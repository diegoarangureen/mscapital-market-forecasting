"""Audit label-free global prediction centering for leading local models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


SOURCE_FILES = {
    "TABM001": "exp-tabm-001_valid.feather",
    "TABM007_GAINPRUNE": "exp-tabm-007-gainprune_valid.feather",
    "XGB053R": "exp-tree-053r_valid.feather",
    "BLEND001_PUBLIC_BEST": "exp-blend-001_valid.feather",
    "BLEND003": "exp-blend-003_valid.feather",
    "BLEND005_ROBUST": "exp-blend-005_valid.feather",
}


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def evaluate(
    target: np.ndarray, prediction: np.ndarray, months: np.ndarray
) -> dict[str, float]:
    month_values = np.sort(np.unique(months))
    monthly_scores = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    primary_months = (month_values >= 62) & (month_values != 66)
    primary_scores = monthly_scores[primary_months]
    return {
        "overall": cosine(target, prediction),
        "no66": cosine(
            target[(months >= 62) & (months != 66)],
            prediction[(months >= 62) & (months != 66)],
        ),
        "recent": cosine(target[months >= 67], prediction[months >= 67]),
        "early": cosine(target[months <= 64], prediction[months <= 64]),
        "primary_std": float(primary_scores.std(ddof=0)),
        "primary_worst": float(primary_scores.min()),
        "primary_q25": float(np.quantile(primary_scores, 0.25)),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["TABM001"]
    validation_rows = reference[["sample_id", "month", "target"]]
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    rows = []
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        raw_prediction = frame["prediction"].to_numpy(dtype=np.float64)
        centered_prediction = raw_prediction - raw_prediction.mean()
        raw = evaluate(target, raw_prediction, months)
        centered = evaluate(target, centered_prediction, months)
        row = {
            "model": name,
            "prediction_mean": float(raw_prediction.mean()),
            "prediction_std": float(raw_prediction.std(ddof=0)),
        }
        for metric_name in raw:
            row[f"raw_{metric_name}"] = raw[metric_name]
            row[f"centered_{metric_name}"] = centered[metric_name]
            row[f"change_{metric_name}"] = centered[metric_name] - raw[metric_name]
        rows.append(row)

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-POST-001-CENTERING-AUDIT"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "global_centering_results.csv", index=False)
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "method": "subtract each prediction vector's own global mean",
                "uses_labels_for_transformation": False,
                "source_files": SOURCE_FILES,
                "deployment_note": (
                    "if selected, subtract the full test prediction mean before submission"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    columns = [
        "model",
        "prediction_mean",
        "raw_overall",
        "centered_overall",
        "change_overall",
        "raw_no66",
        "centered_no66",
        "change_no66",
        "raw_recent",
        "centered_recent",
        "change_recent",
        "raw_primary_worst",
        "centered_primary_worst",
        "change_primary_worst",
    ]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()
