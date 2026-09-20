"""Evaluate fixed-LR quantile TabM at every epoch on three forward windows."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_011_quantile_preprocessing import (
    QuantileBatchPreprocessor,
    evaluate,
    fit_quantile_knots,
)


EXPERIMENT_ID = "EXP-TABM-014-QUANTILE-STAGE-CURVES"
MAX_EPOCHS = 20
FOLDS = {
    "train039_valid4049": (39, 40, 49),
    "train049_valid5059": (49, 50, 59),
    "train059_valid6070": (59, 60, 70),
}


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0.0:
        raise AssertionError("Prediction norm must be finite and non-zero.")
    return values / norm


def load_xgboost_prediction(
    project_dir: Path, fold_name: str, validation_ids: np.ndarray
) -> np.ndarray:
    if fold_name == "train039_valid4049":
        frame = pd.read_feather(
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-012-EARLY-FORWARD-PREPROCESSING"
            / "xgboost_predictions.feather"
        )
        prediction = frame["prediction"].to_numpy(dtype=np.float64)
        if len(prediction) != len(validation_ids):
            raise AssertionError("Early XGBoost row count differs.")
        return prediction
    if fold_name == "train049_valid5059":
        frame = pd.read_feather(
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
            / "validation_predictions.feather"
        )
        column = "xgboost"
    else:
        frame = pd.read_feather(
            project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
        )
        column = "prediction"
    if not np.array_equal(frame["sample_id"].to_numpy(), validation_ids):
        raise AssertionError(f"XGBoost IDs differ for {fold_name}.")
    return frame[column].to_numpy(dtype=np.float64)


def run_fold(
    project_dir: Path,
    fold_name: str,
    train_end: int,
    valid_start: int,
    valid_end: int,
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    sample_ids: np.ndarray,
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    curve_path = fold_dir / "epoch_curve.csv"
    result_path = fold_dir / "result.json"
    if curve_path.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    train_indices = np.flatnonzero(months <= train_end)
    valid_indices = np.flatnonzero(
        (months >= valid_start) & (months <= valid_end)
    )
    knots, medians, missing_columns = fit_quantile_knots(
        features, train_indices, tabm.SEED
    )
    preprocessor = QuantileBatchPreprocessor(
        knots, medians, missing_columns, device
    )
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_std).astype(np.float32)
    validation_target = target[valid_indices].astype(np.float64)
    validation_months = months[valid_indices]
    validation_ids = sample_ids[valid_indices]
    xgboost = load_xgboost_prediction(project_dir, fold_name, validation_ids)
    xgboost_unit = unit(xgboost)

    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = tabm.train_one_epoch(
            model,
            optimizer,
            features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        prediction = tabm.predict(model, features, valid_indices, preprocessor)
        prediction = prediction.astype(np.float64) * target_std
        blend = 0.75 * unit(prediction) + 0.25 * xgboost_unit
        tabm_metrics = evaluate(
            validation_target,
            prediction,
            validation_months,
            valid_start,
            valid_end,
        )
        blend_metrics = evaluate(
            validation_target,
            blend,
            validation_months,
            valid_start,
            valid_end,
        )
        row = {
            "epoch": epoch,
            "gradient_steps_approx": int(
                epoch * np.ceil(len(train_indices) / tabm.BATCH_SIZE)
            ),
            "train_mse": float(loss),
            **{
                f"tabm_{key}": value
                for key, value in tabm_metrics.items()
                if key not in {"monthly", "first_half_months", "second_half_months"}
            },
            **{
                f"b001_{key}": value
                for key, value in blend_metrics.items()
                if key not in {"monthly", "first_half_months", "second_half_months"}
            },
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(curve_path, index=False)
        print(
            f"{fold_name} epoch={epoch:02d} tabm={tabm_metrics['overall']:.8f} "
            f"b001={blend_metrics['overall']:.8f}",
            flush=True,
        )

    curve = pd.DataFrame(rows)
    best_tabm = curve.loc[curve["tabm_overall"].idxmax()].to_dict()
    best_b001 = curve.loc[curve["b001_overall"].idxmax()].to_dict()
    result = {
        "fold": fold_name,
        "train_months": f"0-{train_end}",
        "validation_months": f"{valid_start}-{valid_end}",
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(valid_indices)),
        "learning_rate": tabm.LEARNING_RATE,
        "learning_rate_schedule": "constant",
        "best_tabm": best_tabm,
        "best_b001": best_b001,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, optimizer, preprocessor, knots, medians
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    tabm.SEED = 42
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, feature_columns, _, _ = tabm.load_exp053r_data(project_dir)
    features = model_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    del model_data
    gc.collect()

    results = {}
    for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
        results[fold_name] = run_fold(
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
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "purpose": "fixed-LR epoch and approximate gradient-step transfer audit",
                "max_epochs": MAX_EPOCHS,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
