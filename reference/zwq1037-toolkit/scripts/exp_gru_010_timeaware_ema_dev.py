"""Paired raw-versus-EMA test for the owned time-aware three-stream GRU."""

from __future__ import annotations

import ast
import gc
import json
import math
import os
import random
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v9_factorized_gru.py"
GRID = ROOT / "data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid"
STATIC_PATH = ROOT / "data/interim/our379_reference_cache/features.npy"
CACHE_META = ROOT / "data/interim/kaggle_relative319_dev"
RUN = ROOT / "data/interim/tree_experiments/EXP-GRU-010-TIMEAWARE-EMA"
FOLD = RUN / "train059_valid6270_ex66"

SEED = 2026
EPOCHS = 6
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LR = 2e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
MARKET_LEN = 200
FLOW_LEN = 60
MARKET_SECONDS = 600.0
FLOW_SECONDS = 60.0
MARKET_FEATURES = list(range(11))
TX_FEATURES = list(range(7))
ORDER_FEATURES = list(range(10))
STATIC_FEATURE_COUNT = 379


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine(target, prediction) -> float:
    target = np.asarray(target, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(target @ prediction / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30))


def extract_runtime(static):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            definitions[node.name] = ast.unparse(node)
    norm_literal = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BUNDLED_STREAM_NORM" for target in node.targets)
    )
    namespace = {
        "np": np, "torch": torch, "nn": nn, "F": F, "math": math, "Dataset": Dataset,
        "SEED": SEED, "MARKET_FEATURES": MARKET_FEATURES, "TX_FEATURES": TX_FEATURES,
        "ORDER_FEATURES": ORDER_FEATURES, "MARKET_LEN": MARKET_LEN, "FLOW_LEN": FLOW_LEN,
        "MARKET_SECONDS": MARKET_SECONDS, "FLOW_SECONDS": FLOW_SECONDS,
        "STATIC_FEATURE_COUNT": STATIC_FEATURE_COUNT, "_STATIC_FEATURES": static,
        "_STATIC_NORM": None,
    }
    for name in ["_compute_static_norm", "GridDataset", "cosine_loss", "_TimeAwareGRUStreamEncoder", "_JointMultiStreamStaticModel"]:
        exec(definitions[name], namespace)
    norm = {
        name: {key: np.asarray(value, np.float32) for key, value in values.items()}
        for name, values in norm_literal.items()
    }
    return namespace, norm


def open_arrays(rows: int):
    specs = {
        "market": ("train_v2_market_200x11.mmap", (rows, 200, 11)),
        "tx": ("train_v2_tx_60x7.mmap", (rows, 60, 7)),
        "order": ("train_v2_order_60x10.mmap", (rows, 60, 10)),
    }
    arrays = {}
    for name, (filename, shape) in specs.items():
        path = GRID / filename
        expected = int(np.prod(shape)) * np.dtype(np.float16).itemsize
        if path.stat().st_size != expected:
            raise AssertionError(f"Unexpected cache size: {path}")
        arrays[name] = np.memmap(path, mode="r", dtype=np.float16, shape=shape)
    return arrays


@torch.no_grad()
def predict(model, loader, device, target_scale):
    model.eval()
    output = []
    for batch in loader:
        inputs = batch[:4]
        prediction = model(*[tensor.to(device, non_blocking=True).contiguous() for tensor in inputs])
        output.append(torch.nan_to_num(prediction).float().cpu().numpy())
    return np.concatenate(output).astype(np.float64) * target_scale


def swap_to_ema(model, ema_parameters):
    backup = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            backup[name] = parameter.detach().clone()
            parameter.copy_(ema_parameters[name])
    return backup


def restore_parameters(model, backup):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(backup[name])


def scores(target, months, prediction):
    masks = {
        "all_ex66": months != 66,
        "62_65": (months >= 62) & (months <= 65),
        "67_70": (months >= 67) & (months <= 70),
    }
    result = {name: cosine(target[mask], prediction[mask]) for name, mask in masks.items()}
    result["monthly"] = {
        str(month): cosine(target[months == month], prediction[months == month])
        for month in [62, 63, 64, 65, 67, 68, 69, 70]
    }
    return result


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    FOLD.mkdir(parents=True, exist_ok=True)
    ids = np.load(CACHE_META / "sample_ids.npy", mmap_mode="r")
    months = np.load(CACHE_META / "months.npy", mmap_mode="r")
    target = np.load(CACHE_META / "targets.npy", mmap_mode="r")
    static = np.load(STATIC_PATH, mmap_mode="r")
    if static.shape != (len(ids), STATIC_FEATURE_COUNT):
        raise AssertionError(f"Unexpected static shape: {static.shape}")
    arrays = open_arrays(len(ids))
    train_indices = np.flatnonzero(months <= 59)
    valid_indices = np.flatnonzero((months >= 62) & (months <= 70))
    target_scale = float(np.std(target[train_indices]))
    namespace, norm = extract_runtime(static)
    namespace["_STATIC_NORM"] = namespace["_compute_static_norm"](static, train_indices)
    dataset_class = namespace["GridDataset"]
    train_dataset = dataset_class(arrays, train_indices, norm, target=target, target_scale=target_scale)
    valid_dataset = dataset_class(arrays, valid_indices, norm, target=target, target_scale=target_scale)
    generator = torch.Generator().manual_seed(SEED + 17)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    seed_all(SEED)
    device = torch.device("cuda")
    model = namespace["_JointMultiStreamStaticModel"]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    history = []
    best = {
        "raw": {"selection": -1e9},
        "ema": {"selection": -1e9},
    }
    truth = np.asarray(target[valid_indices], np.float64)
    valid_months = np.asarray(months[valid_indices])
    start_epoch = 1
    recovery_path = FOLD / "recovery_checkpoint.pt"
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location=device, weights_only=False)
        completed_epoch = int(recovery["epoch"])
        if completed_epoch < EPOCHS:
            model.load_state_dict(recovery["model"])
            optimizer.load_state_dict(recovery["optimizer"])
            scheduler.load_state_dict(recovery["scheduler"])
            ema_parameters = {name: value.to(device) for name, value in recovery["ema_parameters"].items()}
            history = recovery["history"]
            generator.set_state(recovery["generator_state"].cpu())
            for name in ("raw", "ema"):
                earlier_best = max(row[name]["62_65"] for row in history)
                if history[-1][name]["62_65"] + 1e-12 < earlier_best:
                    raise RuntimeError("Recovery checkpoint is not the selection-best epoch; cannot reconstruct prior best state")
            raw_prediction = predict(model, valid_loader, device, target_scale)
            backup = swap_to_ema(model, ema_parameters)
            ema_prediction = predict(model, valid_loader, device, target_scale)
            ema_state = cpu_state(model)
            restore_parameters(model, backup)
            raw_scores = scores(truth, valid_months, raw_prediction)
            ema_scores = scores(truth, valid_months, ema_prediction)
            best["raw"] = {
                "selection": raw_scores["62_65"], "epoch": completed_epoch,
                "scores": raw_scores, "prediction": raw_prediction.copy(),
                "state": cpu_state(model),
            }
            best["ema"] = {
                "selection": ema_scores["62_65"], "epoch": completed_epoch,
                "scores": ema_scores, "prediction": ema_prediction.copy(),
                "state": ema_state,
            }
            start_epoch = completed_epoch + 1
            seed_all(SEED + 1000 + start_epoch)
            print(f"resuming from epoch {completed_epoch}", flush=True)
    started = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            market, transaction, order, static_batch, batch_target = batch
            inputs = [tensor.to(device, non_blocking=True).contiguous() for tensor in (market, transaction, order, static_batch)]
            batch_target = batch_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = torch.nan_to_num(model(*inputs))
            loss = 0.35 * F.smooth_l1_loss(prediction, batch_target) + 0.65 * namespace["cosine_loss"](prediction, batch_target)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    ema_parameters[name].lerp_(parameter.detach(), 1.0 - EMA_DECAY)
            total += float(loss.detach()) * len(batch_target)
            count += len(batch_target)
        scheduler.step()
        raw_prediction = predict(model, valid_loader, device, target_scale)
        backup = swap_to_ema(model, ema_parameters)
        ema_prediction = predict(model, valid_loader, device, target_scale)
        restore_parameters(model, backup)
        truth = np.asarray(target[valid_indices], np.float64)
        valid_months = np.asarray(months[valid_indices])
        raw_scores = scores(truth, valid_months, raw_prediction)
        ema_scores = scores(truth, valid_months, ema_prediction)
        row = {
            "epoch": epoch, "loss": total / max(count, 1),
            "raw": raw_scores, "ema": ema_scores,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        for name, prediction_values, metric_values in (
            ("raw", raw_prediction, raw_scores), ("ema", ema_prediction, ema_scores)
        ):
            if metric_values["62_65"] > best[name]["selection"]:
                if name == "raw":
                    state = cpu_state(model)
                else:
                    swap_backup = swap_to_ema(model, ema_parameters)
                    state = cpu_state(model)
                    restore_parameters(model, swap_backup)
                best[name] = {
                    "selection": metric_values["62_65"], "epoch": epoch,
                    "scores": metric_values, "prediction": prediction_values.copy(),
                    "state": state,
                }
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "ema_parameters": ema_parameters,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "generator_state": generator.get_state(),
        }, FOLD / "recovery_checkpoint.pt")

    raw_best = best["raw"]
    ema_best = best["ema"]
    deltas = {
        key: ema_best["scores"][key] - raw_best["scores"][key]
        for key in ["all_ex66", "62_65", "67_70"]
    }
    passed = bool(deltas["all_ex66"] >= 0.0001 and deltas["62_65"] >= 0 and deltas["67_70"] >= 0)
    pd.DataFrame({
        "sample_id": np.asarray(ids[valid_indices]),
        "month": np.asarray(months[valid_indices]),
        "target": np.asarray(target[valid_indices]),
        "raw": raw_best["prediction"],
        "ema": ema_best["prediction"],
    }).to_feather(FOLD / "validation_predictions.feather")
    torch.save(raw_best["state"], FOLD / "raw_best_state.pt")
    torch.save(ema_best["state"], FOLD / "ema_best_state.pt")
    summary = {
        "experiment": "EXP-GRU-010-TIMEAWARE-EMA",
        "architecture": "three independent time-aware GRU96 streams + static379 + four-token fusion",
        "ema_decay": EMA_DECAY,
        "selection_rule": "choose raw and EMA epochs independently on months62-65",
        "raw_best": {k: v for k, v in raw_best.items() if k not in {"prediction", "state"}},
        "ema_best": {k: v for k, v in ema_best.items() if k not in {"prediction", "state"}},
        "deltas": deltas,
        "pass_gate": "all_ex66>=+0.0001 and 62-65/67-70 deltas>=0",
        "passed": passed,
        "history": history,
    }
    (RUN / "score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


