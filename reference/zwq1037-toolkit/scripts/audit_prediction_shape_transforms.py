"""Audit label-free shape transforms on the centered-component 80/20 blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def evaluate(
    target: np.ndarray,
    prediction: np.ndarray,
    months: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    month_values = np.sort(np.unique(months))
    monthly = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    primary_months = (month_values >= 62) & (month_values != 66)
    primary = monthly[primary_months]
    lomo = []
    for omitted_month in month_values[primary_months]:
        mask = (months >= 62) & (months != 66) & (months != omitted_month)
        lomo.append(
            cosine(target[mask], prediction[mask])
            - cosine(target[mask], baseline[mask])
        )
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
        "primary_lomo_min": float(np.min(lomo)),
        "primary_lomo_mean": float(np.mean(lomo)),
    }


def centered_unit(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    return centered / np.linalg.norm(centered)


def winsorize(values: np.ndarray, fraction: float) -> np.ndarray:
    lower, upper = np.quantile(values, [fraction, 1.0 - fraction])
    return np.clip(values, lower, upper)


def rank_uniform(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks - 0.5) / len(values) - 0.5


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    tabm = pd.read_feather(prediction_dir / "exp-tabm-001_valid.feather")
    tree = pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather")
    validation_rows = tabm[["sample_id", "month", "target"]]
    assert_prediction_alignment(tree, validation_rows)
    target = tabm["target"].to_numpy(dtype=np.float64)
    months = tabm["month"].to_numpy()
    base_prediction = (
        0.80 * centered_unit(tabm["prediction"].to_numpy(dtype=np.float64))
        + 0.20 * centered_unit(tree["prediction"].to_numpy(dtype=np.float64))
    )

    uniform_rank = rank_uniform(base_prediction)
    rank_probability = uniform_rank + 0.5
    rank_probability = np.clip(rank_probability, 1.0e-6, 1.0 - 1.0e-6)
    transformations = {
        "identity": base_prediction,
        "winsor_0.1pct": winsorize(base_prediction, 0.001),
        "winsor_0.5pct": winsorize(base_prediction, 0.005),
        "winsor_1pct": winsorize(base_prediction, 0.01),
        "winsor_2pct": winsorize(base_prediction, 0.02),
        "signed_power_0.75": np.sign(base_prediction) * np.abs(base_prediction) ** 0.75,
        "signed_power_1.25": np.sign(base_prediction) * np.abs(base_prediction) ** 1.25,
        "rank_uniform": uniform_rank,
        "rank_gaussian": ndtri(rank_probability),
    }
    rows = []
    for name, prediction in transformations.items():
        # 所有形状变换后再去均值，部署时只需测试预测自身。
        # Recenter every transformed vector using its own distribution only.
        prediction = prediction - prediction.mean()
        metrics = evaluate(target, prediction, months, base_prediction)
        rows.append({"transformation": name, **metrics})

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-POST-003-SHAPE-TRANSFORMS"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "shape_transform_results.csv", index=False)
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "base": "center each component, normalize, blend 80% TabM + 20% XGBoost",
                "uses_labels_for_transformation": False,
                "selection_policy": (
                    "require overall, no66, recent, worst, q25, and primary LOMO "
                    "to be non-decreasing versus identity"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
