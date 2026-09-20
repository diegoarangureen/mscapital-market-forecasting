"""Validate, register, and select the Kaggle V12 temporal GRU outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = (
    PROJECT
    / "data"
    / "interim"
    / "kaggle_outputs"
    / "multistream_factorized_gru_temporal3fold_v12"
)
RUN_NAME = "standalone_timeaware_three_stream_gru379_temporal3fold"
OUTPUT_DIR = PROJECT / "outputs" / "submissions"


def find_unique(name: str) -> Path:
    matches = list(DOWNLOAD_DIR.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def load_validated(path: Path, expected_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected V12 columns: {path}")
    if len(frame) != 647_896:
        raise AssertionError(f"Unexpected V12 rows in {path}: {len(frame)}")
    if not np.array_equal(frame["sample_id"].to_numpy(), expected_ids):
        raise AssertionError(f"V12 IDs differ from baseline: {path}")
    values = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or values.std() <= 0.0:
        raise AssertionError(f"Invalid V12 predictions: {path}")
    return frame


def main() -> None:
    result_path = find_unique("result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise AssertionError(f"V12 result is not complete: {result.get('status')}")
    if int(result.get("visible_gpu_count", 0)) < 2:
        raise AssertionError("V12 did not use both Kaggle GPUs")

    baseline_path = OUTPUT_DIR / "standalone_timeaware_three_stream_gru379.csv"
    baseline = pd.read_csv(baseline_path).sort_values("sample_id").reset_index(drop=True)
    expected_ids = baseline["sample_id"].to_numpy()
    baseline_values = baseline["prediction"].to_numpy(dtype=np.float64)

    ensemble_source = find_unique("factorized_gru379_temporal3fold_e4.csv")
    ensemble = load_validated(ensemble_source, expected_ids)
    ensemble_output = OUTPUT_DIR / f"{RUN_NAME}.csv"
    ensemble.to_csv(ensemble_output, index=False)

    fold_outputs = {}
    fold_correlations = {}
    for fold_end in result.get("fold_ends", [52, 57, 62]):
        fold_end = int(fold_end)
        fold_source = find_unique(f"factorized_gru379_fold_end{fold_end}_e4.csv")
        fold = load_validated(fold_source, expected_ids)
        fold_output = OUTPUT_DIR / f"standalone_timeaware_three_stream_gru379_fold_end{fold_end}.csv"
        fold.to_csv(fold_output, index=False)
        fold_outputs[str(fold_end)] = str(fold_output)
        fold_correlations[str(fold_end)] = {
            "with_fulltrain_gru": float(
                np.corrcoef(
                    fold["prediction"].to_numpy(dtype=np.float64), baseline_values
                )[0, 1]
            ),
            "with_temporal_equal3": float(
                np.corrcoef(
                    fold["prediction"].to_numpy(dtype=np.float64),
                    ensemble["prediction"].to_numpy(dtype=np.float64),
                )[0, 1]
            ),
        }

    per_fold_validation = {
        str(int(row["fold_end"])): float(row["strict_valid_cosine"])
        for row in result.get("folds", [])
    }
    if len(per_fold_validation) != 3:
        raise AssertionError("V12 strict validation scores are missing")
    best_fold = max(per_fold_validation, key=per_fold_validation.get)
    ensemble_validation = float(result["strict_validation_ensemble_cosine"])
    best_fold_validation = per_fold_validation[best_fold]
    if ensemble_validation >= best_fold_validation:
        recommended_variant = "equal3"
        recommended_path = ensemble_output
        recommended_validation = ensemble_validation
    else:
        recommended_variant = f"fold_end{best_fold}"
        recommended_path = Path(fold_outputs[best_fold])
        recommended_validation = best_fold_validation

    ensemble_values = ensemble["prediction"].to_numpy(dtype=np.float64)
    report = {
        "run_name": RUN_NAME,
        "experiment": result.get("experiment"),
        "fold_ends": result.get("fold_ends"),
        "seeds": result.get("seeds"),
        "model_count": result.get("model_count"),
        "epochs_per_model": result.get("epochs_per_model"),
        "visible_gpu_count": result.get("visible_gpu_count"),
        "gpu_names": result.get("gpu_names"),
        "strict_validation_window": result.get("strict_validation_window"),
        "strict_validation_rows": result.get("strict_validation_rows"),
        "strict_validation_per_fold": per_fold_validation,
        "strict_validation_equal3": ensemble_validation,
        "recommended_variant": recommended_variant,
        "recommended_validation_cosine": recommended_validation,
        "recommended_output_path": str(recommended_path),
        "fold_output_paths": fold_outputs,
        "fold_correlations": fold_correlations,
        "rows": len(ensemble),
        "finite": True,
        "prediction_mean": float(ensemble_values.mean()),
        "prediction_std": float(ensemble_values.std()),
        "correlation_with_fulltrain_gru": float(
            np.corrcoef(ensemble_values, baseline_values)[0, 1]
        ),
        "source_path": str(ensemble_source),
        "output_path": str(ensemble_output),
        "submission_status": "prepared_not_submitted",
        "submission_policy": "screen for final blend; no standalone slot reserved",
    }
    metadata_path = PROJECT / "outputs" / "submission_metadata" / f"{RUN_NAME}.json"
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
