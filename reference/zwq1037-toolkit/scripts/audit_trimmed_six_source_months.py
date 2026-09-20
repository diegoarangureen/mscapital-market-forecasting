"""Export per-month cosine comparisons for trimmed six-source candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit_trimmed_tabm_in_blends import centered_unit, cosine, unit
from prepare_trimmed_six_source_submissions import CANDIDATES


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    reference = pd.read_feather(prediction_dir / "exp-tabm-001-trim1_valid.feather")
    ids = reference["sample_id"].to_numpy()
    cosine_aggregations = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
        / "cosine_aggregations_valid.feather"
    )
    frames = {
        "original": reference,
        "corrprune": pd.read_feather(
            prediction_dir / "exp-tabm-006-corrprune_valid.feather"
        ),
        "cosine": cosine_aggregations,
        "xgboost": pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather"),
        "lightgbm": pd.read_feather(prediction_dir / "exp-tree-068_valid.feather"),
        "histgb": pd.read_feather(
            prediction_dir / "hist_gradient_boosting_target_centered_level2_valid.feather"
        ),
        "public_best": pd.read_feather(prediction_dir / "exp-blend-001_valid.feather"),
    }
    for name, frame in frames.items():
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError(f"Validation IDs are not aligned for {name}.")

    sources = {
        "original": centered_unit(
            frames["original"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "corrprune": unit(
            frames["corrprune"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "cosine": unit(
            frames["cosine"]["trim1_prediction"].to_numpy(dtype=np.float64)
        ),
        "xgboost": centered_unit(
            frames["xgboost"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "lightgbm": centered_unit(
            frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64)
        ),
        "histgb": unit(
            frames["histgb"]["prediction"].to_numpy(dtype=np.float64)
        ),
    }
    predictions = {
        name: sum(weights[source] * sources[source] for source in sources)
        for name, weights in CANDIDATES.items()
    }
    predictions["public_best_b001"] = frames["public_best"]["prediction"].to_numpy(
        dtype=np.float64
    )
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    rows = []
    for month in np.sort(np.unique(months)):
        mask = months == month
        row = {"month": int(month), "rows": int(mask.sum())}
        for name, prediction in predictions.items():
            row[name] = cosine(target[mask], prediction[mask])
        rows.append(row)
    monthly = pd.DataFrame(rows)
    for name in CANDIDATES:
        monthly[f"{name}_change_vs_b001"] = (
            monthly[name] - monthly["public_best_b001"]
        )
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-013-SIX-SOURCE-MONTHS"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(run_dir / "monthly_comparison.csv", index=False)
    print(monthly.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
