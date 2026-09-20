"""Strict nested forward backtest of B001 on train 0-49, validate 50-59."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

import exp_tabm_001_exp053r_features as tabm
from audit_tabm_member_aggregation import predict_members
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters


EXPERIMENT_ID = "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
INNER_TRAIN_END = 39
INNER_VALID_START = 40
INNER_VALID_END = 49
FORMAL_TRAIN_END = 49
FORMAL_VALID_START = 50
FORMAL_VALID_END = 59


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def evaluate(target: np.ndarray, prediction: np.ndarray, months: np.ndarray) -> dict:
    month_values = np.sort(np.unique(months))
    monthly = [
        {
            "month": int(month),
            "rows": int((months == month).sum()),
            "cosine": cosine(target[months == month], prediction[months == month]),
        }
        for month in month_values
    ]
    monthly_values = np.asarray([row["cosine"] for row in monthly])
    return {
        "overall": cosine(target, prediction),
        "first_half_50_54": cosine(target[months <= 54], prediction[months <= 54]),
        "second_half_55_59": cosine(target[months >= 55], prediction[months >= 55]),
        "monthly_macro_mean": float(monthly_values.mean()),
        "monthly_std": float(monthly_values.std(ddof=0)),
        "monthly_worst": float(monthly_values.min()),
        "monthly_q25": float(np.quantile(monthly_values, 0.25)),
        "monthly": monthly,
    }


def fit_preprocessing(
    features: np.ndarray, indices: np.ndarray, device: torch.device
) -> tuple[tabm.BatchPreprocessor, tuple[np.ndarray, ...]]:
    arrays = tabm.fit_preprocessor(features, indices)
    return tabm.BatchPreprocessor(*arrays, device), arrays


def scaled_targets(
    target: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, float, float]:
    mean = float(target[indices].mean(dtype=np.float64))
    standard_deviation = float(target[indices].std(dtype=np.float64))
    scaled = ((target - mean) / standard_deviation).astype(np.float32)
    return scaled, mean, standard_deviation


def select_epoch(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    device: torch.device,
) -> tuple[int, list[dict[str, float]], float]:
    train_indices = np.flatnonzero(months <= INNER_TRAIN_END)
    valid_indices = np.flatnonzero(
        (months >= INNER_VALID_START) & (months <= INNER_VALID_END)
    )
    preprocessor, _ = fit_preprocessing(features, train_indices, device)
    target_scaled, _, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    best_epoch = 1
    best_score = -np.inf
    stale_epochs = 0
    logs: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, tabm.MAX_SCOUT_EPOCHS + 1):
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
        prediction *= target_std
        score = cosine(target[valid_indices], prediction)
        if score > best_score + tabm.MIN_DELTA:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        logs.append(
            {
                "epoch": epoch,
                "train_mse": float(loss),
                "cosine_40_49": score,
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
        )
        print(
            f"inner epoch={epoch:02d} cosine_40_49={score:.8f} "
            f"best={best_score:.8f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= tabm.PATIENCE:
            break
    elapsed = time.perf_counter() - started
    del model, optimizer, preprocessor
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, logs, elapsed


def train_formal_tabm(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    feature_columns: list[str],
    best_epoch: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_indices = np.flatnonzero(months <= FORMAL_TRAIN_END)
    valid_indices = np.flatnonzero(
        (months >= FORMAL_VALID_START) & (months <= FORMAL_VALID_END)
    )
    preprocessor, preprocessing_arrays = fit_preprocessing(features, train_indices, device)
    target_scaled, target_mean, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    logs = []
    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        loss = tabm.train_one_epoch(
            model,
            optimizer,
            features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        logs.append({"epoch": epoch, "train_mse": float(loss)})
        print(f"formal TabM epoch={epoch:02d}/{best_epoch:02d} loss={loss:.8f}", flush=True)
    elapsed = time.perf_counter() - started
    validation_members = predict_members(
        model,
        features[valid_indices],
        preprocessor,
        target_std,
    )
    mean_prediction = validation_members.mean(axis=1)
    sorted_members = np.sort(validation_members, axis=1)
    trim1_prediction = sorted_members[:, 1:-1].mean(axis=1)
    model_cpu = model.cpu()
    torch.save(
        {
            "state_dict": model_cpu.state_dict(),
            "input_dimension": preprocessor.output_dimension,
            "feature_columns": feature_columns,
            "target_mean": target_mean,
            "target_standard_deviation": target_std,
            "best_epoch": best_epoch,
            "seed": tabm.SEED,
        },
        run_dir / "tabm_model.pt",
    )
    np.savez_compressed(
        run_dir / "tabm_preprocessing.npz",
        medians=preprocessing_arrays[0],
        means=preprocessing_arrays[1],
        standard_deviations=preprocessing_arrays[2],
        missing_columns=preprocessing_arrays[3],
        feature_columns=np.asarray(feature_columns),
    )
    pd.DataFrame(logs).to_csv(run_dir / "tabm_training.csv", index=False)
    details = {
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "target_mean": target_mean,
        "target_standard_deviation": target_std,
        "member_count": int(validation_members.shape[1]),
    }
    del model_cpu, model, optimizer, preprocessor, validation_members
    torch.cuda.empty_cache()
    gc.collect()
    return mean_prediction, trim1_prediction, details


def train_xgboost(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    train_indices = np.flatnonzero(months <= FORMAL_TRAIN_END)
    valid_indices = np.flatnonzero(
        (months >= FORMAL_VALID_START) & (months <= FORMAL_VALID_END)
    )
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    centered_target = target[train_indices] - target_mean
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    parameters["device"] = "cuda"
    parameters["n_jobs"] = 2
    model = XGBRegressor(**parameters)
    started = time.perf_counter()
    model.fit(features[train_indices], centered_target)
    elapsed = time.perf_counter() - started
    prediction = np.asarray(model.predict(features[valid_indices]), dtype=np.float64)
    model.save_model(run_dir / "xgboost_model.json")
    details = {
        "training_seconds": elapsed,
        "target_mean": target_mean,
        "parameters": parameters,
    }
    del model
    gc.collect()
    return prediction, details


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
    months = model_data["month"].to_numpy()
    validation_mask = (months >= FORMAL_VALID_START) & (months <= FORMAL_VALID_END)
    validation_rows = model_data.loc[
        validation_mask, ["sample_id", "month", "target"]
    ].reset_index(drop=True)
    del model_data
    gc.collect()

    best_epoch, inner_logs, inner_seconds = select_epoch(
        features, target, months, torch.device("cuda")
    )
    pd.DataFrame(inner_logs).to_csv(run_dir / "inner_epoch_selection.csv", index=False)
    tabm_mean, tabm_trim1, tabm_details = train_formal_tabm(
        features,
        target,
        months,
        feature_columns,
        best_epoch,
        torch.device("cuda"),
        run_dir,
    )
    xgboost_prediction, xgboost_details = train_xgboost(
        features, target, months, run_dir
    )
    del features
    gc.collect()

    b001 = (
        0.75 * tabm_mean / np.linalg.norm(tabm_mean)
        + 0.25 * xgboost_prediction / np.linalg.norm(xgboost_prediction)
    )
    b001_trim1 = (
        0.75 * tabm_trim1 / np.linalg.norm(tabm_trim1)
        + 0.25 * xgboost_prediction / np.linalg.norm(xgboost_prediction)
    )
    validation_target = validation_rows["target"].to_numpy(dtype=np.float64)
    validation_months = validation_rows["month"].to_numpy()
    predictions = {
        "tabm_mean": tabm_mean,
        "tabm_trim1": tabm_trim1,
        "xgboost": xgboost_prediction,
        "b001_mean": b001,
        "b001_trim1": b001_trim1,
    }
    metrics = {
        name: evaluate(validation_target, prediction, validation_months)
        for name, prediction in predictions.items()
    }
    output = validation_rows.copy()
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_feather(run_dir / "validation_predictions.feather")
    monthly = pd.DataFrame(
        {
            "month": list(range(FORMAL_VALID_START, FORMAL_VALID_END + 1)),
            **{
                name: [row["cosine"] for row in values["monthly"]]
                for name, values in metrics.items()
            },
        }
    )
    monthly.to_csv(run_dir / "monthly_cosine.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": {
            "inner_train": "0-39",
            "inner_validation": "40-49",
            "formal_train": "0-49",
            "formal_validation": "50-59",
            "features_and_blend_weights": "frozen from B001",
            "no_overlap": True,
        },
        "feature_count": len(feature_columns),
        "train_rows": int((months <= FORMAL_TRAIN_END).sum()),
        "validation_rows": int(validation_mask.sum()),
        "tabm_epoch_selection": {
            "selected_epoch": best_epoch,
            "elapsed_seconds": inner_seconds,
            "logs": inner_logs,
        },
        "tabm": tabm_details,
        "xgboost": xgboost_details,
        "metrics": metrics,
        "prediction_path": str(run_dir / "validation_predictions.feather"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
