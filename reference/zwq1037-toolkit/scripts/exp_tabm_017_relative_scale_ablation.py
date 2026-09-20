"""Run one group-level ablation of EXP-TABM-016 relative-scale features."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_014_quantile_stage_curves import FOLDS
from exp_tabm_016_relative_scale_features import (
    DERIVED_RELATIVE_COLUMNS,
    MAX_EPOCHS,
    PRICE_RELATIVE_COLUMNS,
    add_relative_features,
    run_fold,
)


VARIANTS = {
    "price8": {
        "experiment_id": "EXP-TABM-017-RELATIVE-PRICE-ADD8",
        "columns": PRICE_RELATIVE_COLUMNS,
        "description": "add only eight within-sample relative price/spread features",
    },
    "activity4": {
        "experiment_id": "EXP-TABM-018-RELATIVE-ACTIVITY-ADD4",
        "columns": DERIVED_RELATIVE_COLUMNS,
        "description": "add only four within-sample relative activity features",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = VARIANTS[args.variant]
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    tabm.SEED = 42
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / spec["experiment_id"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, baseline_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, model_data)
    candidate_columns = [*baseline_columns, *spec["columns"]]
    if len(candidate_columns) != len(set(candidate_columns)):
        raise AssertionError("Candidate feature schema contains duplicates.")
    features = model_data[candidate_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    del model_data
    gc.collect()

    fold_results = {}
    for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
        fold_results[fold_name] = run_fold(
            project_dir,
            fold_name,
            train_end,
            valid_start,
            valid_end,
            features,
            target,
            months,
            sample_ids,
            run_dir,
            torch.device("cuda"),
        )
    result = {
        "experiment_id": spec["experiment_id"],
        "single_variable_change": spec["description"],
        "baseline": "EXP-TABM-015 epoch 15, same seed/model/quantile/LR policy",
        "baseline_feature_count": len(baseline_columns),
        "candidate_feature_count": len(candidate_columns),
        "added_features": spec["columns"],
        "epochs": MAX_EPOCHS,
        "folds": fold_results,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
