"""Cross the 6/16 epoch budgets for standard and quantile preprocessing."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_011_quantile_preprocessing import evaluate, train_formal as train_quantile
from exp_tabm_012_early_forward_preprocessing import (
    CONFIG,
    train_standard_formal as train_standard,
    unit,
)


EXPERIMENT_ID = "EXP-TABM-013-PREPROCESSING-BUDGET-CROSSCHECK"


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    tabm.SEED = 42

    project_dir = Path(__file__).resolve().parents[1]
    source_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-012-EARLY-FORWARD-PREPROCESSING"
    )
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    standard16_dir = run_dir / "standard_epoch16"
    quantile6_dir = run_dir / "quantile_epoch6"
    standard16_dir.mkdir(parents=True, exist_ok=True)
    quantile6_dir.mkdir(parents=True, exist_ok=True)

    model_data, feature_columns, _, _ = tabm.load_exp053r_data(project_dir)
    features = model_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    del model_data
    gc.collect()

    standard16_mean, _ = train_standard(
        features,
        target,
        months,
        16,
        torch.device("cuda"),
        standard16_dir,
    )
    quantile6_mean, quantile6_trim1, _ = train_quantile(
        features,
        target,
        months,
        CONFIG,
        6,
        torch.device("cuda"),
        quantile6_dir,
    )
    standard6 = pd.read_feather(source_dir / "standard_predictions.feather")
    quantile16 = pd.read_feather(
        source_dir / "quantile" / "quantile_predictions.feather"
    )
    xgboost = pd.read_feather(source_dir / "xgboost_predictions.feather")[
        "prediction"
    ].to_numpy(dtype=np.float64)
    valid_mask = (
        (months >= CONFIG["formal_valid_start"])
        & (months <= CONFIG["formal_valid_end"])
    )
    validation_target = target[valid_mask].astype(np.float64)
    validation_months = months[valid_mask]
    raw_predictions = {
        "standard_epoch6": standard6["mean_prediction"].to_numpy(dtype=np.float64),
        "standard_epoch16": standard16_mean,
        "quantile_epoch6": quantile6_mean,
        "quantile_epoch16": quantile16["mean_prediction"].to_numpy(dtype=np.float64),
    }
    predictions = dict(raw_predictions)
    for name, prediction in raw_predictions.items():
        predictions[f"{name}_b001"] = 0.75 * unit(prediction) + 0.25 * unit(xgboost)
    metrics = {
        name: evaluate(
            validation_target,
            prediction,
            validation_months,
            CONFIG["formal_valid_start"],
            CONFIG["formal_valid_end"],
        )
        for name, prediction in predictions.items()
    }
    summary = pd.DataFrame(
        [
            {
                "candidate": name,
                **{key: value for key, value in values.items() if key != "monthly"},
            }
            for name, values in metrics.items()
        ]
    ).sort_values("overall", ascending=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    output = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "separate preprocessing effect from the 6 versus 16 epoch budget",
        "formal_train": "0-39",
        "formal_validation": "40-49",
        "learning_rate": tabm.LEARNING_RATE,
        "learning_rate_schedule": "constant",
        "metrics": metrics,
    }
    (run_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
