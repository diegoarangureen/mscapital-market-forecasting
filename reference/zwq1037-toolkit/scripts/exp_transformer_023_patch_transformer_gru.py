"""Controlled structured-patch Transformer-GRU test against EXP020 raw."""

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
SOURCE = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v7_baseline.py"
GRID = ROOT / "data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid"
STATIC_PATH = ROOT / "data/interim/our379_reference_cache/features.npy"
CACHE_META = ROOT / "data/interim/kaggle_relative319_dev"
RUN = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-023-PATCH-GRU"
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
    for name in ["_compute_static_norm", "GridDataset", "cosine_loss", "ConvBlock", "_FactorizedStreamEncoder", "_JointMultiStreamStaticModel"]:
        exec(definitions[name], namespace)
    norm = {
        name: {key: np.asarray(value, np.float32) for key, value in values.items()}
        for name, values in norm_literal.items()
    }
    return namespace, norm



class PatchTransformerGRUStream(nn.Module):
    """Encode short structured temporal patches, then restore recurrence with a GRU."""

    def __init__(
        self,
        input_dim: int,
        length: int,
        patch_size: int = 4,
        d_model: int = 96,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if length % patch_size != 0:
            raise ValueError(f"length={length} must be divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.patch_count = length // patch_size
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * patch_size),
            nn.Linear(input_dim * patch_size, d_model),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.recurrent_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, feature_count = values.shape
        patches = values.reshape(
            batch_size,
            self.patch_count,
            self.patch_size * feature_count,
        )
        patch_mask = patches.abs().sum(dim=-1) == 0
        tokens = self.projection(patches) + self.position
        tokens = self.encoder(tokens, src_key_padding_mask=patch_mask)
        recurrent, _ = self.gru(tokens.masked_fill(patch_mask.unsqueeze(-1), 0.0))
        gate = self.recurrent_gate(torch.cat([tokens, recurrent], dim=-1))
        tokens = gate * recurrent + (1.0 - gate) * tokens
        logits = self.attention(tokens).squeeze(-1).masked_fill(patch_mask, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)


class PatchTransformerGRUModel(nn.Module):
    """Apply the same patch-recurrent encoder independently to three streams."""

    def __init__(self, d_model: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.market = PatchTransformerGRUStream(
            len(MARKET_FEATURES), MARKET_LEN, patch_size=4, d_model=d_model, dropout=dropout
        )
        self.transaction = PatchTransformerGRUStream(
            len(TX_FEATURES), FLOW_LEN, patch_size=4, d_model=d_model, dropout=dropout
        )
        self.order = PatchTransformerGRUStream(
            len(ORDER_FEATURES), FLOW_LEN, patch_size=4, d_model=d_model, dropout=dropout
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, d_model),
            nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.source_attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(
        self,
        market: torch.Tensor,
        transaction: torch.Tensor,
        order: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        summaries = torch.stack(
            [
                self.market(market),
                self.transaction(transaction),
                self.order(order),
                self.static_adapter(static),
            ],
            dim=1,
        )
        summaries = self.fusion(summaries + self.source_embedding)
        weights = torch.softmax(
            self.source_attention(summaries).squeeze(-1), dim=1
        ).unsqueeze(-1)
        return self.head((summaries * weights).sum(dim=1)).squeeze(-1)


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
    model = PatchTransformerGRUModel().to(device)
    print(f"parameter_count={sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    history = []
    best = {
        "raw": {"selection": -1e9},
        "ema": {"selection": -1e9},
    }
    checkpoint_path = FOLD / "recovery_checkpoint.pt"
    start_epoch = 1
    started = time.time()
    if checkpoint_path.exists():
        recovery = torch.load(checkpoint_path, map_location=device, weights_only=False)
        recovered_epoch = int(recovery["epoch"])
        if recovered_epoch < EPOCHS:
            model.load_state_dict(recovery["model"])
            ema_parameters = {
                name: value.to(device)
                for name, value in recovery["ema_parameters"].items()
            }
            optimizer.load_state_dict(recovery["optimizer"])
            scheduler.load_state_dict(recovery["scheduler"])
            history = recovery["history"]
            # torch.load(map_location=device) can move the CPU generator state
            # onto CUDA. Generator.set_state requires a CPU ByteTensor.
            generator_state = recovery["generator_state"]
            if torch.is_tensor(generator_state):
                generator_state = generator_state.detach().cpu().to(torch.uint8)
            else:
                generator_state = torch.as_tensor(
                    generator_state, dtype=torch.uint8, device="cpu"
                )
            generator.set_state(generator_state)
            if "best" in recovery:
                best = recovery["best"]
            else:
                # Epoch 4 is the best raw and EMA checkpoint so far, so both
                # historical best states can be reconstructed exactly.
                truth = np.asarray(target[valid_indices], np.float64)
                valid_months = np.asarray(months[valid_indices])
                raw_prediction = predict(model, valid_loader, device, target_scale)
                backup = swap_to_ema(model, ema_parameters)
                ema_prediction = predict(model, valid_loader, device, target_scale)
                ema_state = cpu_state(model)
                restore_parameters(model, backup)
                raw_scores = scores(truth, valid_months, raw_prediction)
                ema_scores = scores(truth, valid_months, ema_prediction)
                for name, prediction_values, metric_values, state in (
                    ("raw", raw_prediction, raw_scores, cpu_state(model)),
                    ("ema", ema_prediction, ema_scores, ema_state),
                ):
                    historical_best = max(
                        row[name]["62_65"] for row in history
                    )
                    if metric_values["62_65"] + 1e-10 < historical_best:
                        raise RuntimeError(
                            f"Cannot reconstruct historical {name} best state"
                        )
                    best[name] = {
                        "selection": metric_values["62_65"],
                        "epoch": recovered_epoch,
                        "scores": metric_values,
                        "prediction": prediction_values.copy(),
                        "state": state,
                    }
            start_epoch = recovered_epoch + 1
            started = time.time() - float(history[-1]["seconds"])
            torch.manual_seed(SEED + 10000 + recovered_epoch)
            torch.cuda.manual_seed_all(SEED + 10000 + recovered_epoch)
            print(
                f"resume_from_epoch={recovered_epoch}, next_epoch={start_epoch}",
                flush=True,
            )
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
            "best": best,
        }, checkpoint_path)

    raw_best = best["raw"]
    ema_best = best["ema"]
    selected_name = (
        "raw"
        if raw_best["scores"]["62_65"] >= ema_best["scores"]["62_65"]
        else "ema"
    )
    selected = best[selected_name]
    baseline_path = (
        ROOT
        / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA"
        / "train059_valid6270_ex66/validation_predictions.feather"
    )
    baseline_frame = pd.read_feather(baseline_path)
    current_ids = np.asarray(ids[valid_indices])
    current_months = np.asarray(months[valid_indices])
    truth = np.asarray(target[valid_indices], np.float64)
    for name, expected, observed in (
        ("sample_id", current_ids, baseline_frame["sample_id"].to_numpy()),
        ("month", current_months, baseline_frame["month"].to_numpy()),
        ("target", truth, baseline_frame["target"].to_numpy()),
    ):
        if not np.allclose(expected, observed, rtol=0.0, atol=1e-8):
            raise AssertionError(f"EXP020 alignment mismatch: {name}")
    baseline_prediction = baseline_frame["raw"].to_numpy(np.float64)
    baseline_scores = scores(truth, current_months, baseline_prediction)
    deltas = {
        key: selected["scores"][key] - baseline_scores[key]
        for key in ["all_ex66", "62_65", "67_70"]
    }
    monthly_deltas = {
        month: selected["scores"]["monthly"][month] - baseline_scores["monthly"][month]
        for month in baseline_scores["monthly"]
    }
    passed = bool(
        deltas["62_65"] >= 0.0005
        and deltas["67_70"] >= 0.0005
        and deltas["all_ex66"] >= 0.0007
        and min(monthly_deltas.values()) >= -0.002
    )
    pd.DataFrame(
        {
            "sample_id": current_ids,
            "month": current_months,
            "target": truth,
            "baseline_exp020_raw": baseline_prediction,
            "raw": raw_best["prediction"],
            "ema": ema_best["prediction"],
            "candidate": selected["prediction"],
        }
    ).to_feather(FOLD / "validation_predictions.feather")
    torch.save(raw_best["state"], FOLD / "raw_best_state.pt")
    torch.save(ema_best["state"], FOLD / "ema_best_state.pt")
    summary = {
        "experiment": "EXP-TRANSFORMER-023-PATCH-TRANSFORMER-GRU",
        "change": "four-step structured patches, per-stream Transformer, gated GRU recurrence",
        "architecture": "patch projection -> Transformer -> gated GRU per stream + static379 fusion",
        "patch_sizes": {"market": 4, "transaction": 4, "order": 4},
        "ema_decay": EMA_DECAY,
        "baseline": baseline_scores,
        "selected_candidate": selected_name,
        "raw_best": {k: v for k, v in raw_best.items() if k not in {"prediction", "state"}},
        "ema_best": {k: v for k, v in ema_best.items() if k not in {"prediction", "state"}},
        "deltas": deltas,
        "monthly_deltas": monthly_deltas,
        "pass_gate": "vs EXP020 raw: selection>=+0.0005; forward>=+0.0005; all>=+0.0007; worst monthly delta>=-0.002",
        "passed": passed,
        "history": history,
    }
    (RUN / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

