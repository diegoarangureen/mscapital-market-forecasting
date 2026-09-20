"""Paired EMA test on the fixed TabM385-k32 training trajectory."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata


PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / "data/interim/kaggle_kernels/relative319_xs_tabm_notebook"
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_extracted as recipe
import exp_gru_003_strong_joint as gru
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_COLUMNS


EXPERIMENT_ID = "EXP-TABM-045-MONTH-RANK-SHRINK"
RUN_DIR = PROJECT / "data/interim/tree_experiments" / EXPERIMENT_ID
FOLD_DIR = RUN_DIR / "train059_valid6270_ex66"
TRAJECTORY_COLUMNS = (
    "mid_return_180 mid_return_60 mid_return_20 "
    "realized_volatility_20 realized_volatility_60 mid_momentum_60"
).split()
EMA_DECAY = 0.999


def make_model_k32(input_dimension: int, device: torch.device):
    return recipe.TabM.make(
        n_num_features=input_dimension,
        cat_cardinalities=None,
        d_out=1,
        k=32,
        n_blocks=2,
        d_block=256,
        dropout=0.1,
        arch_type="tabm",
    ).to(device)


def paired_metrics(target, months, raw_prediction, ema_prediction):
    raw = recipe.evaluate(target, raw_prediction, months)
    ema = recipe.evaluate(target, ema_prediction, months)
    masks = {
        "62_65": (months >= 62) & (months <= 65),
        "67_70": (months >= 67) & (months <= 70),
    }
    halves = {}
    for name, mask in masks.items():
        raw_score = recipe.cosine(target[mask], raw_prediction[mask])
        ema_score = recipe.cosine(target[mask], ema_prediction[mask])
        halves[name] = {"raw": raw_score, "ema": ema_score, "delta": ema_score - raw_score}
    overall_delta = ema["months_62_70_without_66"] - raw["months_62_70_without_66"]
    passed = bool(
        overall_delta >= 0.0001
        and halves["62_65"]["delta"] >= 0.0
        and halves["67_70"]["delta"] >= 0.0
    )
    return raw, ema, halves, overall_delta, passed


def train_paired(features, target, months, device):
    train_indices = np.flatnonzero(months <= 59)
    valid_indices = np.flatnonzero((months >= 62) & (months <= 70))
    knots, medians, missing_columns = recipe.fit_quantile_knots(features, train_indices)
    preprocessor = recipe.QuantilePreprocessor(knots, medians, missing_columns, device)
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    scaled_target = ((target - target_mean) / target_std).astype(np.float32)
    training_months = months[train_indices]
    ranked_target = np.empty(len(train_indices), dtype=np.float32)
    for month in np.unique(training_months):
        local = np.flatnonzero(training_months == month)
        ranked = (2.0 * rankdata(target[train_indices][local], method="average") / (len(local) + 1.0) - 1.0).astype(np.float32)
        ranked = (ranked - ranked.mean()) / max(float(ranked.std()), 1e-8)
        ranked_target[local] = ranked
    scaled_target[train_indices] = 0.75 * scaled_target[train_indices] + 0.25 * ranked_target
    recipe.seed_everything(recipe.SEED)
    model = make_model_k32(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.LEARNING_RATE, weight_decay=recipe.WEIGHT_DECAY
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ema_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    logs = []
    started = time.perf_counter()
    for epoch in range(1, recipe.EPOCHS + 1):
        rate = recipe.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        shuffled = np.random.default_rng(recipe.SEED + epoch).permutation(train_indices)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(shuffled), recipe.BATCH_SIZE):
            batch_indices = shuffled[start:start + recipe.BATCH_SIZE]
            batch = preprocessor.transform(features[batch_indices])
            batch_target = torch.as_tensor(
                scaled_target[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                member_predictions = model(batch).squeeze(-1)
                loss = F.mse_loss(
                    member_predictions.float(),
                    batch_target[:, None].expand_as(member_predictions),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in ema_parameters:
                        ema_parameters[name].lerp_(parameter.detach(), 1.0 - EMA_DECAY)
            total_loss += float(loss.detach().cpu()) * len(batch_indices)
            total_count += len(batch_indices)
        row = {"epoch": epoch, "learning_rate": rate, "train_mse": total_loss / total_count}
        logs.append(row)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_parameters": ema_parameters,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "logs": logs,
            },
            FOLD_DIR / "recovery_checkpoint.pt",
        )
        print(json.dumps(row), flush=True)

    raw_prediction = recipe.predict(model, features, valid_indices, preprocessor, amp_dtype)
    raw_prediction = raw_prediction.astype(np.float64) * target_std
    raw_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in ema_parameters:
                parameter.copy_(ema_parameters[name])
    ema_prediction = recipe.predict(model, features, valid_indices, preprocessor, amp_dtype)
    ema_prediction = ema_prediction.astype(np.float64) * target_std
    ema_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return {
        "valid_indices": valid_indices,
        "raw_prediction": raw_prediction,
        "ema_prediction": ema_prediction,
        "raw_state": raw_state,
        "ema_state": ema_state,
        "logs": logs,
        "seconds": time.perf_counter() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_feather(
        PROJECT / "data/raw/label.feather", columns=["sample_id", "month", "target"]
    ).sort_values("sample_id").reset_index(drop=True)
    base, base_columns = gru.load_relative319(PROJECT, labels)
    sample_ids = labels.sample_id.to_numpy(copy=True)
    months = labels.month.to_numpy(dtype=np.int16, copy=True)
    target = labels.target.to_numpy(dtype=np.float32, copy=True)
    xs40, xs_columns = recipe.add_relative_features(base, months, base_columns)
    trajectory = pd.read_feather(
        PROJECT / "data/processed/train_market_microstructure_features.feather",
        columns=["sample_id", *TRAJECTORY_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    order = pd.read_feather(
        PROJECT / "data/processed/train_order_quote_position_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(sample_ids, trajectory.sample_id.to_numpy()):
        raise AssertionError("Trajectory IDs do not align")
    if not np.array_equal(sample_ids, order.sample_id.to_numpy()):
        raise AssertionError("Order IDs do not align")
    features = np.concatenate([
        base,
        xs40,
        order[ORDER_COLUMNS].to_numpy(dtype=np.float32, copy=True),
        trajectory[TRAJECTORY_COLUMNS].to_numpy(dtype=np.float32, copy=True),
    ], axis=1)
    if features.shape != (len(labels), 385):
        raise AssertionError(f"Unexpected features: {features.shape}")
    del labels, trajectory, order, base, xs40
    gc.collect()

    trained = train_paired(features, target, months, torch.device("cuda"))
    valid = trained["valid_indices"]
    raw, ema, halves, overall_delta, passed = paired_metrics(
        target[valid].astype(np.float64), months[valid],
        trained["raw_prediction"], trained["ema_prediction"],
    )
    predictions = pd.DataFrame({
        "sample_id": sample_ids[valid],
        "month": months[valid],
        "target": target[valid],
        "raw": trained["raw_prediction"],
        "ema": trained["ema_prediction"],
    })
    predictions.to_feather(FOLD_DIR / "validation_predictions.feather")
    torch.save(trained["raw_state"], FOLD_DIR / "raw_final_state.pt")
    torch.save(trained["ema_state"], FOLD_DIR / "ema_final_state.pt")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "single_variable_change": "25pct month-rank target plus 75pct raw-z target; EMA paired control",
        "features": 385,
        "k": 32,
        "epochs": recipe.EPOCHS,
        "seed": recipe.SEED,
        "parameter_count": trained["parameter_count"],
        "seconds": trained["seconds"],
        "raw_metrics": raw,
        "ema_metrics": ema,
        "paired_halves": halves,
        "overall_delta": overall_delta,
        "pass_gate": "overall>=+0.0001 and both 62-65/67-70 deltas>=0",
        "passed": passed,
        "logs": trained["logs"],
    }
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

