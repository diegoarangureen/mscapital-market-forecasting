"""Third nested forward gate for standard versus quantile TabM preprocessing."""

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
from exp_tabm_011_quantile_preprocessing import (
    evaluate,
    select_epoch as select_quantile_epoch,
    train_formal as train_quantile_formal,
)
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters


EXPERIMENT_ID = "EXP-TABM-012-EARLY-FORWARD-PREPROCESSING"
CONFIG = {
    "inner_train_end": 29,
    "inner_valid_start": 30,
    "inner_valid_end": 39,
    "formal_train_end": 39,
    "formal_valid_start": 40,
    "formal_valid_end": 49,
}


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0.0:
        raise AssertionError("Prediction norm must be finite and non-zero.")
    return values / norm


def fit_standard_preprocessor(
    features: np.ndarray, indices: np.ndarray, device: torch.device
) -> tabm.BatchPreprocessor:
    return tabm.BatchPreprocessor(*tabm.fit_preprocessor(features, indices), device)


def scaled_targets(target: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, float]:
    mean = float(target[indices].mean(dtype=np.float64))
    standard_deviation = float(target[indices].std(dtype=np.float64))
    scaled = ((target - mean) / standard_deviation).astype(np.float32)
    return scaled, standard_deviation


def select_standard_epoch(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    device: torch.device,
    run_dir: Path,
) -> tuple[int, list[dict[str, float]]]:
    result_path = run_dir / "standard_epoch_selection.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        return int(saved["best_epoch"]), saved["logs"]
    train_indices = np.flatnonzero(months <= CONFIG["inner_train_end"])
    valid_indices = np.flatnonzero(
        (months >= CONFIG["inner_valid_start"])
        & (months <= CONFIG["inner_valid_end"])
    )
    preprocessor = fit_standard_preprocessor(features, train_indices, device)
    target_scaled, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    best_epoch = 1
    best_score = -np.inf
    stale_epochs = 0
    logs = []
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
                "validation_cosine": score,
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
        )
        print(
            f"standard inner epoch={epoch:02d} cosine={score:.8f} "
            f"best={best_score:.8f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= tabm.PATIENCE:
            break
    result_path.write_text(
        json.dumps(
            {"best_epoch": best_epoch, "best_score": best_score, "logs": logs},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, optimizer, preprocessor
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, logs


def train_standard_formal(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    best_epoch: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    output_path = run_dir / "standard_predictions.feather"
    if output_path.exists():
        frame = pd.read_feather(output_path)
        return (
            frame["mean_prediction"].to_numpy(dtype=np.float64),
            frame["trim1_prediction"].to_numpy(dtype=np.float64),
        )
    train_indices = np.flatnonzero(months <= CONFIG["formal_train_end"])
    valid_indices = np.flatnonzero(
        (months >= CONFIG["formal_valid_start"])
        & (months <= CONFIG["formal_valid_end"])
    )
    preprocessor = fit_standard_preprocessor(features, train_indices, device)
    target_scaled, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
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
        print(
            f"standard formal epoch={epoch:02d}/{best_epoch:02d} loss={loss:.8f}",
            flush=True,
        )
    members = predict_members(model, features[valid_indices], preprocessor, target_std)
    mean_prediction = members.mean(axis=1).astype(np.float64)
    trim1_prediction = np.sort(members, axis=1)[:, 1:-1].mean(axis=1).astype(
        np.float64
    )
    pd.DataFrame(
        {
            "mean_prediction": mean_prediction,
            "trim1_prediction": trim1_prediction,
        }
    ).to_feather(output_path)
    del model, optimizer, preprocessor, members
    torch.cuda.empty_cache()
    gc.collect()
    return mean_prediction, trim1_prediction


def train_xgboost(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    run_dir: Path,
) -> np.ndarray:
    output_path = run_dir / "xgboost_predictions.feather"
    if output_path.exists():
        return pd.read_feather(output_path)["prediction"].to_numpy(dtype=np.float64)
    train_indices = np.flatnonzero(months <= CONFIG["formal_train_end"])
    valid_indices = np.flatnonzero(
        (months >= CONFIG["formal_valid_start"])
        & (months <= CONFIG["formal_valid_end"])
    )
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    parameters["device"] = "cuda"
    parameters["n_jobs"] = 2
    model = XGBRegressor(**parameters)
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    started = time.perf_counter()
    model.fit(features[train_indices], target[train_indices] - target_mean)
    prediction = np.asarray(model.predict(features[valid_indices]), dtype=np.float64)
    pd.DataFrame({"prediction": prediction}).to_feather(output_path)
    model.save_model(run_dir / "xgboost_model.json")
    print(f"xgboost seconds={time.perf_counter() - started:.2f}", flush=True)
    del model
    gc.collect()
    return prediction


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

    standard_epoch, standard_logs = select_standard_epoch(
        features, target, months, torch.device("cuda"), run_dir
    )
    standard_mean, standard_trim1 = train_standard_formal(
        features,
        target,
        months,
        standard_epoch,
        torch.device("cuda"),
        run_dir,
    )
    quantile_dir = run_dir / "quantile"
    quantile_dir.mkdir(parents=True, exist_ok=True)
    quantile_epoch, quantile_logs, _, = select_quantile_epoch(
        features,
        target,
        months,
        CONFIG,
        torch.device("cuda"),
        quantile_dir,
    )
    quantile_mean, quantile_trim1, quantile_details = train_quantile_formal(
        features,
        target,
        months,
        CONFIG,
        quantile_epoch,
        torch.device("cuda"),
        quantile_dir,
    )
    xgboost_prediction = train_xgboost(features, target, months, run_dir)

    valid_mask = (
        (months >= CONFIG["formal_valid_start"])
        & (months <= CONFIG["formal_valid_end"])
    )
    validation = pd.DataFrame(
        {
            "sample_id": sample_ids[valid_mask],
            "month": months[valid_mask],
            "target": target[valid_mask],
        }
    )
    predictions = {
        "standard_tabm_mean": standard_mean,
        "quantile_tabm_mean": quantile_mean,
        "quantile_tabm_trim1": quantile_trim1,
        "standard_b001": 0.75 * unit(standard_mean) + 0.25 * unit(xgboost_prediction),
        "quantile_b001_mean": 0.75 * unit(quantile_mean)
        + 0.25 * unit(xgboost_prediction),
        "quantile_b001_trim1": 0.75 * unit(quantile_trim1)
        + 0.25 * unit(xgboost_prediction),
    }
    validation_target = validation["target"].to_numpy(dtype=np.float64)
    validation_months = validation["month"].to_numpy()
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
    standard_tabm_score = metrics["standard_tabm_mean"]["overall"]
    standard_b001_score = metrics["standard_b001"]["overall"]
    summary = pd.DataFrame(
        [
            {
                "candidate": name,
                **{key: value for key, value in values.items() if key != "monthly"},
                "change_vs_matching_baseline": values["overall"]
                - (
                    standard_b001_score
                    if "b001" in name
                    else standard_tabm_score
                ),
            }
            for name, values in metrics.items()
        ]
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    monthly = pd.DataFrame(
        {
            "month": list(range(40, 50)),
            **{
                name: [row["cosine"] for row in values["monthly"]]
                for name, values in metrics.items()
            },
        }
    )
    monthly.to_csv(run_dir / "monthly_cosine.csv", index=False)
    output = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": CONFIG,
        "standard_selected_epoch": standard_epoch,
        "quantile_selected_epoch": quantile_epoch,
        "standard_inner_logs": standard_logs,
        "quantile_inner_logs": quantile_logs,
        "quantile_formal": quantile_details,
        "metrics": metrics,
    }
    (run_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
