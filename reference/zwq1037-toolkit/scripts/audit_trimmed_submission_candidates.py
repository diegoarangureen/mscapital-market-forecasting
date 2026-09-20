"""Validate prepared trimmed-TabM submissions and compare their test geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = {
    "public_best_b001": "tabm001_tree053r_blend75_fulltrain.csv",
    "gainprune_277": "tabm007_gainprune_fulltrain.csv",
    "centered80": "tabm001_tree053r_centered_components_blend80_fulltrain.csv",
    "aggressive_mean": "tabm_four_source_45_20_20_xgb15_fulltrain.csv",
    "guarded_mean": "tabm_four_source_guarded_50_25_05_xgb20_fulltrain.csv",
    "aggressive_trim1": "tabm_trim1_four_source_aggressive_45_20_20_xgb15_fulltrain.csv",
    "guarded_trim1": "tabm_trim1_four_source_guarded_50_25_05_xgb20_fulltrain.csv",
}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    expected_ids = template["sample_id"].to_numpy()
    submission_dir = project_dir / "outputs" / "submissions"

    predictions: dict[str, np.ndarray] = {}
    integrity: dict[str, object] = {}
    for name, filename in CANDIDATES.items():
        frame = pd.read_csv(submission_dir / filename)
        values = frame["prediction"].to_numpy(dtype=np.float64)
        valid = (
            list(frame.columns) == ["sample_id", "prediction"]
            and len(frame) == len(template)
            and np.array_equal(frame["sample_id"].to_numpy(), expected_ids)
            and not frame["sample_id"].duplicated().any()
            and np.isfinite(values).all()
        )
        if not valid:
            raise AssertionError(f"Integrity check failed for {name}: {filename}")
        predictions[name] = values
        integrity[name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "finite": True,
            "ids_exactly_match_template": True,
            "duplicate_ids": 0,
        }

    reference = predictions["public_best_b001"]
    reference_norm = float(np.linalg.norm(reference))
    comparisons: dict[str, object] = {}
    for name, values in predictions.items():
        difference = values - reference
        comparisons[name] = {
            "correlation_vs_public_best": float(np.corrcoef(values, reference)[0, 1]),
            "relative_l2_change_vs_public_best": float(
                np.linalg.norm(difference) / reference_norm
            ),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    paired_changes: dict[str, object] = {}
    for prefix in ["aggressive", "guarded"]:
        mean_values = predictions[f"{prefix}_mean"]
        trim_values = predictions[f"{prefix}_trim1"]
        paired_changes[prefix] = {
            "correlation_trim_vs_mean": float(np.corrcoef(trim_values, mean_values)[0, 1]),
            "relative_l2_change_trim_vs_mean": float(
                np.linalg.norm(trim_values - mean_values) / np.linalg.norm(mean_values)
            ),
        }

    result = {
        "template_rows": int(len(template)),
        "integrity": integrity,
        "comparisons": comparisons,
        "paired_changes": paired_changes,
    }
    output_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-003-SUBMISSION-AUDIT"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
