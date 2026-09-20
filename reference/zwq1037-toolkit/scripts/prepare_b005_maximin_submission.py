"""Prepare the maximin month-balanced 5% interpolation from trimmed B005."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_NAME = "tabm_double_trim_b005_maximin_alpha005_fulltrain"
ALPHA_ROBUST = 0.05


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    submission_dir = project_dir / "outputs" / "submissions"
    prediction_dir = project_dir / "outputs" / "predictions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    b005 = pd.read_csv(submission_dir / "tabm_double_trim_blend005_fulltrain.csv")
    robust = pd.read_csv(
        submission_dir
        / "tabm_double_trim_six_robust_40_20_20_05_05_10_fulltrain.csv"
    )
    for name, frame in [("b005", b005), ("robust", robust)]:
        if not np.array_equal(
            frame["sample_id"].to_numpy(), template["sample_id"].to_numpy()
        ):
            raise AssertionError(f"{name} IDs do not match submission template.")
    prediction = (
        (1.0 - ALPHA_ROBUST) * b005["prediction"].to_numpy(dtype=np.float64)
        + ALPHA_ROBUST * robust["prediction"].to_numpy(dtype=np.float64)
    )
    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    if (
        not np.isfinite(prediction).all()
        or submission["sample_id"].duplicated().any()
        or submission.isna().any().any()
    ):
        raise AssertionError("Balanced submission contains invalid values.")

    interpolation = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-014-B005-ROBUST-INTERPOLATION"
        / "interpolation_grid.csv"
    )
    local_row = interpolation.loc[
        np.isclose(interpolation["alpha_robust"], ALPHA_ROBUST)
    ]
    if len(local_row) != 1:
        raise AssertionError("Could not locate alpha=0.05 validation result.")
    local_row = local_row.iloc[0]
    weights = {
        name.removeprefix("weight_"): float(local_row[name])
        for name in local_row.index
        if name.startswith("weight_")
    }
    metric_names = [
        "overall",
        "no66",
        "recent",
        "early",
        "primary_std",
        "primary_worst",
        "primary_q25",
        "primary_months_improved",
        "primary_month_min_change",
        "primary_lomo_min",
        "primary_lomo_mean",
        "month65_change",
        "month69_change",
    ]
    local_metrics = {
        name: (
            int(local_row[name])
            if name == "primary_months_improved"
            else float(local_row[name])
        )
        for name in metric_names
    }
    output_path = submission_dir / f"{OUTPUT_NAME}.csv"
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    submission.to_csv(output_path, index=False)
    submission.to_feather(prediction_path)
    metadata = {
        "output_name": OUTPUT_NAME,
        "method": "95% trimmed B005 weights + 5% robust-grid weights",
        "selection_objective": (
            "maximize the minimum primary-month change versus public B001 "
            "along the predeclared one-dimensional interpolation"
        ),
        "alpha_robust": ALPHA_ROBUST,
        "weights": weights,
        "training_months": "0-70 for every trained source",
        "aggregation_uses_labels": False,
        "local_validation": local_metrics,
        "test_rows": int(len(submission)),
        "prediction_summary": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std(ddof=0)),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "l2_norm": float(np.linalg.norm(prediction)),
        },
        "submission_path": str(output_path),
        "prediction_path": str(prediction_path),
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
