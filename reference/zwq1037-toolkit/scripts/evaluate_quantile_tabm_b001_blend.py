"""Replace only B001's TabM branch with EXP-TABM-011 predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import FOLDS, cosine, evaluate


EXPERIMENT_ID = "EXP-BLEND-015-QUANTILE-B001"
TABM_WEIGHT = 0.75
TREE_WEIGHT = 0.25


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0.0:
        raise AssertionError("Prediction norm must be finite and non-zero.")
    return values / norm


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    quantile_dir = (
        project_dir / "data" / "interim" / "tree_experiments" / "EXP-TABM-011-QUANTILE"
    )
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for fold_name, config in FOLDS.items():
        comparison = pd.read_feather(
            quantile_dir / fold_name / "comparison_predictions.feather"
        )
        if fold_name == "train049_valid5059":
            tree_frame = pd.read_feather(
                project_dir
                / "data"
                / "interim"
                / "tree_experiments"
                / "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
                / "validation_predictions.feather"
            )
            tree_prediction = tree_frame["xgboost"].to_numpy(dtype=np.float64)
        else:
            tree_frame = pd.read_feather(
                project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
            )
            tree_prediction = tree_frame["prediction"].to_numpy(dtype=np.float64)
        if not np.array_equal(
            comparison["sample_id"].to_numpy(), tree_frame["sample_id"].to_numpy()
        ):
            raise AssertionError(f"Tree rows do not align for {fold_name}.")

        tree_unit = unit(tree_prediction)
        sources = {
            "standard_mean": comparison["baseline_standardized_tabm"].to_numpy(
                dtype=np.float64
            ),
            "quantile_mean": comparison["quantile_tabm_mean"].to_numpy(dtype=np.float64),
            "quantile_trim1": comparison["quantile_tabm_trim1"].to_numpy(
                dtype=np.float64
            ),
        }
        predictions = {
            name: TABM_WEIGHT * unit(values) + TREE_WEIGHT * tree_unit
            for name, values in sources.items()
        }
        target = comparison["target"].to_numpy(dtype=np.float64)
        months = comparison["month"].to_numpy()
        metrics = {
            name: evaluate(
                target,
                prediction,
                months,
                config["formal_valid_start"],
                config["formal_valid_end"],
            )
            for name, prediction in predictions.items()
        }
        baseline = metrics["standard_mean"]["overall"]
        summary = pd.DataFrame(
            [
                {
                    "candidate": name,
                    **{key: value for key, value in values.items() if key != "monthly"},
                    "overall_change_vs_b001": values["overall"] - baseline,
                }
                for name, values in metrics.items()
            ]
        )
        fold_dir = run_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(fold_dir / "summary.csv", index=False)
        monthly = pd.DataFrame(
            {
                "month": list(
                    range(config["formal_valid_start"], config["formal_valid_end"] + 1)
                ),
                **{
                    name: [row["cosine"] for row in values["monthly"]]
                    for name, values in metrics.items()
                },
            }
        )
        monthly.to_csv(fold_dir / "monthly_cosine.csv", index=False)
        results[fold_name] = {"metrics": metrics}
        print(f"\n{fold_name}", flush=True)
        print(summary.to_string(index=False), flush=True)

    output = {
        "experiment_id": EXPERIMENT_ID,
        "single_change_vs_b001": "replace standardized TabM with quantile TabM",
        "weights": {"tabm": TABM_WEIGHT, "xgboost": TREE_WEIGHT},
        "normalization": "per-source validation-vector L2, identical to B001",
        "folds": results,
    }
    (run_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
