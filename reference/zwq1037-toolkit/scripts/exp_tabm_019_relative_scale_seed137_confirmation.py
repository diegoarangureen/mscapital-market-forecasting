"""Paired seed-137 confirmation of 307 baseline versus relative add12."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_016_relative_scale_features import (
    MAX_EPOCHS,
    RELATIVE_COLUMNS,
    add_relative_features,
    run_fold,
)


EXPERIMENT_ID = "EXP-TABM-019-RELATIVE-ADD12-SEED137-PAIRED"
SEED = 137
FOLDS = {
    "train049_valid5059": (49, 50, 59),
    "train059_valid6070": (59, 60, 70),
}


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    tabm.SEED = SEED
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, baseline_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, model_data)
    candidate_columns = [*baseline_columns, *RELATIVE_COLUMNS]
    all_features = model_data[candidate_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    del model_data
    gc.collect()

    results = {}
    for variant, feature_count in (("baseline307", 307), ("relative_add12", 319)):
        variant_dir = run_dir / variant
        variant_results = {}
        variant_features = all_features[:, :feature_count]
        for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
            variant_results[fold_name] = run_fold(
                project_dir,
                fold_name,
                train_end,
                valid_start,
                valid_end,
                variant_features,
                target,
                months,
                sample_ids,
                variant_dir,
                torch.device("cuda"),
            )
        results[variant] = variant_results

    paired = {}
    for fold_name in FOLDS:
        baseline_metrics = results["baseline307"][fold_name]["metrics"]
        candidate_metrics = results["relative_add12"][fold_name]["metrics"]
        fold_comparison = {}
        for prediction_name in ("tabm_mean", "tabm_trim1", "b001_mean", "b001_trim1"):
            baseline_values = baseline_metrics[prediction_name]
            candidate_values = candidate_metrics[prediction_name]
            keys = ["overall", "monthly_std", "monthly_worst", "monthly_q25"]
            if "without_month_66" in candidate_values:
                keys.extend(["without_month_66", "months_67_70"])
            fold_comparison[prediction_name] = {
                key: {
                    "baseline307": baseline_values[key],
                    "relative_add12": candidate_values[key],
                    "delta": candidate_values[key] - baseline_values[key],
                }
                for key in keys
            }
        paired[fold_name] = fold_comparison

    result = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "paired second-seed confirmation of the 12-feature union",
        "seed": SEED,
        "epochs": MAX_EPOCHS,
        "fixed_policy": "quantile preprocessing plus 20-step cosine schedule stopped at epoch 15",
        "added_features": RELATIVE_COLUMNS,
        "results": results,
        "paired_comparison": paired,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(paired, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
