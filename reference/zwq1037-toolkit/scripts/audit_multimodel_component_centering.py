"""Audit component centering for the robust six-source EXP-BLEND-005."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


SOURCE_FILES = {
    "tabm_original": "exp-tabm-001_valid.feather",
    "tabm_corrprune": "exp-tabm-006-corrprune_valid.feather",
    "tabm_cosine": "exp-tabm-003-cosine_valid.feather",
    "xgboost": "exp-tree-053r_valid.feather",
    "lightgbm": "exp-tree-068_valid.feather",
    "histgb": "hist_gradient_boosting_target_centered_level2_valid.feather",
}
BLEND_WEIGHTS = {
    "tabm_original": 0.35,
    "tabm_corrprune": 0.20,
    "tabm_cosine": 0.15,
    "xgboost": 0.10,
    "lightgbm": 0.10,
    "histgb": 0.10,
}


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def evaluate(target: np.ndarray, prediction: np.ndarray, months: np.ndarray) -> dict:
    month_values = np.sort(np.unique(months))
    monthly = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    primary = monthly[(month_values >= 62) & (month_values != 66)]
    return {
        "overall": cosine(target, prediction),
        "no66": cosine(
            target[(months >= 62) & (months != 66)],
            prediction[(months >= 62) & (months != 66)],
        ),
        "recent": cosine(target[months >= 67], prediction[months >= 67]),
        "early": cosine(target[months <= 64], prediction[months <= 64]),
        "primary_std": float(primary.std(ddof=0)),
        "primary_worst": float(primary.min()),
        "primary_q25": float(np.quantile(primary, 0.25)),
    }


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values)


def centered_unit(values: np.ndarray) -> np.ndarray:
    values = values - values.mean()
    return values / np.linalg.norm(values)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_original"]
    validation_rows = reference[["sample_id", "month", "target"]]
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    raw_units = {}
    centered_units = {}
    rows = []
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        values = frame["prediction"].to_numpy(dtype=np.float64)
        raw_units[name] = unit(values)
        centered_units[name] = centered_unit(values)
        raw_metrics = evaluate(target, raw_units[name], months)
        centered_metrics = evaluate(target, centered_units[name], months)
        row = {
            "source": name,
            "prediction_mean": float(values.mean()),
            "prediction_std": float(values.std(ddof=0)),
        }
        for metric in raw_metrics:
            row[f"raw_{metric}"] = raw_metrics[metric]
            row[f"centered_{metric}"] = centered_metrics[metric]
            row[f"change_{metric}"] = centered_metrics[metric] - raw_metrics[metric]
        rows.append(row)

    raw_blend = sum(BLEND_WEIGHTS[name] * raw_units[name] for name in BLEND_WEIGHTS)
    centered_blend = sum(
        BLEND_WEIGHTS[name] * centered_units[name] for name in BLEND_WEIGHTS
    )
    blend_rows = []
    for name, prediction in [
        ("raw_normalized_blend005", raw_blend),
        ("component_centered_blend005", centered_blend),
    ]:
        blend_rows.append({"variant": name, **evaluate(target, prediction, months)})

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-POST-004-MULTIMODEL-CENTERING"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    source_results = pd.DataFrame(rows)
    blend_results = pd.DataFrame(blend_rows)
    source_results.to_csv(run_dir / "source_centering_results.csv", index=False)
    blend_results.to_csv(run_dir / "blend_centering_results.csv", index=False)
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "source_files": SOURCE_FILES,
                "weights": BLEND_WEIGHTS,
                "method": "subtract each source global mean before source L2 normalization",
                "uses_labels_for_transformation": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(source_results[[
        "source", "prediction_mean", "change_overall", "change_no66",
        "change_recent", "change_primary_worst", "change_primary_q25"
    ]].to_string(index=False))
    print(blend_results.to_string(index=False))


if __name__ == "__main__":
    main()
