"""Strict local dev for a lightweight three-stream TSMixer."""

from __future__ import annotations

import ast
import json
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid"
STATIC = ROOT / "data/interim/our379_reference_cache/features.npy"
SOURCE = ROOT / "data/interim/kaggle_relative319_dev"
BASELINE_TRANSFORMER = ROOT / "outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv"
RUN = ROOT / "data/interim/tree_experiments/EXP-SEQUENCE-021-TSMIXER-DEV"
FOLD = RUN / "train059_valid6270_ex66"

ROWS = 1_257_637
SEED = 2026
EPOCHS = 6
BATCH_SIZE = 192
EVAL_BATCH_SIZE = 384
LR = 2e-4
WEIGHT_DECAY = 1e-4
COSINE_WEIGHT = 0.65
NAMES = ["tabm", "realmlp", "transformer", "gru"]
OWNED4_WEIGHTS = np.asarray(
    [0.31819564837516784, 0.13030480191317811,
     0.19215218382533478, 0.35934736588631944], dtype=np.float64
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(target @ prediction / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30))


def centered_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float() - prediction.float().mean()
    target = target.float() - target.float().mean()
    return 1.0 - F.cosine_similarity(prediction, target, dim=0, eps=1e-6)


def load_bundled_norm() -> dict[str, dict[str, np.ndarray]]:
    source = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v25_last_readout.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    value = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BUNDLED_STREAM_NORM" for target in node.targets)
    )
    return {
        name: {key: np.asarray(array, dtype=np.float32) for key, array in stats.items()}
        for name, stats in value.items()
    }


def open_arrays() -> dict[str, np.memmap]:
    specifications = {
        "market": ("train_v2_market_200x11.mmap", (ROWS, 200, 11)),
        "tx": ("train_v2_tx_60x7.mmap", (ROWS, 60, 7)),
        "order": ("train_v2_order_60x10.mmap", (ROWS, 60, 10)),
    }
    arrays = {}
    for name, (filename, shape) in specifications.items():
        path = GRID / filename
        expected = int(np.prod(shape)) * np.dtype(np.float16).itemsize
        if path.stat().st_size != expected:
            raise AssertionError(f"Unexpected mmap size: {path}")
        arrays[name] = np.memmap(path, mode="r", dtype=np.float16, shape=shape)
    return arrays


def static_norm(features: np.memmap, train_indices: np.ndarray) -> dict[str, np.ndarray]:
    path = RUN / "static_norm.npz"
    if path.exists():
        state = np.load(path)
        return {"mean": state["mean"], "std": state["std"]}
    rng = np.random.default_rng(SEED + 91)
    selected = np.sort(rng.choice(train_indices, min(100_000, len(train_indices)), replace=False))
    total = np.zeros(features.shape[1], dtype=np.float64)
    total_sq = np.zeros(features.shape[1], dtype=np.float64)
    count = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(selected), 4096):
        values = np.asarray(features[selected[start:start + 4096]], dtype=np.float32)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        total += safe.sum(0)
        total_sq += (safe * safe).sum(0)
        count += finite.sum(0)
    denominator = np.maximum(count, 1)
    mean = total / denominator
    std = np.sqrt(np.maximum(total_sq / denominator - mean * mean, 0))
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    np.savez(path, mean=mean, std=std)
    return {"mean": mean, "std": std}


class SequenceDataset(Dataset):
    def __init__(self, arrays, static, indices, norm, static_stats, target, target_scale):
        self.arrays = arrays
        self.static = static
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm = norm
        self.static_stats = static_stats
        self.target = target
        self.target_scale = target_scale

    def __len__(self):
        return len(self.indices)

    def sequence(self, name, index):
        values = np.asarray(self.arrays[name][index], dtype=np.float32)
        padding = np.abs(values).sum(-1) == 0
        values = (values - self.norm[name]["mean"]) / self.norm[name]["std"]
        values = np.clip(np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0), -8, 8)
        values[padding] = 0.0
        return values.astype(np.float32, copy=False)

    def __getitem__(self, item):
        index = int(self.indices[item])
        static = np.asarray(self.static[index], dtype=np.float32)
        static = (static - self.static_stats["mean"]) / self.static_stats["std"]
        static = np.clip(np.nan_to_num(static, nan=0.0, posinf=8.0, neginf=-8.0), -8, 8).astype(np.float32)
        values = (
            torch.from_numpy(self.sequence("market", index)),
            torch.from_numpy(self.sequence("tx", index)),
            torch.from_numpy(self.sequence("order", index)),
            torch.from_numpy(static),
            torch.tensor(np.float32(self.target[index] / self.target_scale)),
        )
        return values


class MixerBlock(nn.Module):
    def __init__(self, length: int, width: int, dropout: float):
        super().__init__()
        time_hidden = max(16, length * 2)
        self.time_norm = nn.LayerNorm(width)
        self.time_mlp = nn.Sequential(
            nn.Linear(length, time_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(time_hidden, length), nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm(width)
        self.feature_mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout),
        )

    def forward(self, tokens, valid):
        mixed = self.time_mlp(self.time_norm(tokens).transpose(1, 2)).transpose(1, 2)
        tokens = (tokens + mixed) * valid.unsqueeze(-1)
        tokens = (tokens + self.feature_mlp(self.feature_norm(tokens))) * valid.unsqueeze(-1)
        return tokens


class StreamMixer(nn.Module):
    def __init__(self, input_dim: int, raw_length: int, patch: int, width=64, blocks=3, dropout=0.15):
        super().__init__()
        if raw_length % patch:
            raise ValueError("raw length must divide evenly into patches")
        self.patch = patch
        self.length = raw_length // patch
        self.projection = nn.Linear(input_dim * 3, width)
        self.blocks = nn.ModuleList([MixerBlock(self.length, width, dropout) for _ in range(blocks)])
        self.pool = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, values):
        batch, _, channels = values.shape
        grouped = values.reshape(batch, self.length, self.patch, channels)
        present = grouped.abs().sum(dim=(-1, -2)) > 0
        mean = grouped.mean(2)
        std = grouped.float().std(2, unbiased=False).to(grouped.dtype)
        last = grouped[:, :, -1]
        tokens = self.projection(torch.cat([mean, std, last], dim=-1)) * present.unsqueeze(-1)
        for block in self.blocks:
            tokens = block(tokens, present)
        logits = self.pool(tokens).squeeze(-1).masked_fill(~present, -1e4)
        weights = torch.softmax(logits, dim=1) * present
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bl,bld->bd", weights, tokens)


class ThreeStreamTSMixer(nn.Module):
    def __init__(self, width=64, dropout=0.15):
        super().__init__()
        self.market = StreamMixer(11, 200, 5, width, dropout=dropout)
        self.tx = StreamMixer(7, 60, 3, width, dropout=dropout)
        self.order = StreamMixer(10, 60, 3, width, dropout=dropout)
        self.static = nn.Sequential(
            nn.Linear(379, 192), nn.SiLU(), nn.LayerNorm(192), nn.Dropout(dropout),
            nn.Linear(192, width), nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, width))
        self.source_gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(self, market, tx, order, static):
        sources = torch.stack([self.market(market), self.tx(tx), self.order(order), self.static(static)], dim=1)
        sources = sources + self.source_embedding
        weights = torch.softmax(self.source_gate(sources).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head((sources * weights).sum(1)).squeeze(-1)


@torch.no_grad()
def predict(model, loader, device, target_scale):
    model.eval()
    parts = []
    for market, tx, order, static, _ in loader:
        inputs = [value.to(device, non_blocking=True) for value in (market, tx, order, static)]
        parts.append(model(*inputs).float().cpu().numpy() * target_scale)
    return np.concatenate(parts)


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    rows = 0
    for market, tx, order, static, target in loader:
        inputs = [value.to(device, non_blocking=True) for value in (market, tx, order, static)]
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(*inputs)
            loss = (1 - COSINE_WEIGHT) * F.smooth_l1_loss(prediction.float(), target.float()) + COSINE_WEIGHT * centered_cosine_loss(prediction, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(target)
        rows += len(target)
    return total / rows


def fusion_analysis(ids, months, target, candidate):
    tabm = pd.read_feather(
        ROOT / "data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather"
    )[["sample_id", "candidate"]].rename(columns={"candidate": "tabm"})
    real_dir = ROOT / "data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66"
    real = pd.DataFrame({"sample_id": np.load(real_dir / "validation_sample_ids.npy"), "realmlp": np.load(real_dir / "validation_predictions.npy")})
    transformer = pd.read_csv(BASELINE_TRANSFORMER, usecols=["sample_id", "prediction"]).rename(columns={"prediction": "transformer"})
    gru = pd.read_csv(
        ROOT / "data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv",
        usecols=["sample_id", "prediction"],
    ).rename(columns={"prediction": "gru"})
    frame = pd.DataFrame({"sample_id": ids, "month": months, "target": target, "tsmixer": candidate})
    for source in (tabm, real, transformer, gru):
        frame = frame.merge(source, on="sample_id", validate="one_to_one")
    selection = frame.month.le(65).to_numpy()
    forward = frame.month.ge(67).to_numpy()
    old = frame[NAMES].to_numpy(np.float64)
    old_scale = np.sqrt(np.mean(old[selection] ** 2, axis=0))
    old_prediction = (old / old_scale) @ OWNED4_WEIGHTS
    new = frame.tsmixer.to_numpy(np.float64)
    new_scale = np.sqrt(np.mean(new[selection] ** 2))
    new = new / max(new_scale, 1e-12)
    truth = frame.target.to_numpy(np.float64)
    rows = []
    for weight in (0.0, 0.05, 0.10, 0.15):
        prediction = (1 - weight) * old_prediction + weight * new
        rows.append({
            "tsmixer_weight": weight,
            "selection_62_65": cosine(truth[selection], prediction[selection]),
            "forward_67_70": cosine(truth[forward], prediction[forward]),
            "monthly_forward": {str(month): cosine(truth[frame.month.eq(month)], prediction[frame.month.eq(month)]) for month in range(67, 71)},
        })
    chosen = max(rows, key=lambda row: row["selection_62_65"])
    baseline = rows[0]
    monthly_delta = {month: chosen["monthly_forward"][month] - baseline["monthly_forward"][month] for month in baseline["monthly_forward"]}
    chosen["forward_delta"] = chosen["forward_67_70"] - baseline["forward_67_70"]
    chosen["monthly_delta"] = monthly_delta
    chosen["passed"] = (
        chosen["tsmixer_weight"] > 0
        and chosen["forward_delta"] >= 0.0003
        and sum(value > 0 for value in monthly_delta.values()) >= 2
        and min(monthly_delta.values()) >= -0.0005
    )
    return {"rows": rows, "selected_by_62_65": chosen, "old_scale": old_scale.tolist(), "tsmixer_scale": new_scale}


def main() -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    RUN.mkdir(parents=True, exist_ok=True)
    FOLD.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    ids = np.load(SOURCE / "sample_ids.npy", mmap_mode="r")
    months = np.load(SOURCE / "months.npy", mmap_mode="r")
    target = np.load(SOURCE / "targets.npy", mmap_mode="r")
    if len(ids) != ROWS:
        raise AssertionError("Unexpected label rows")
    train_indices = np.flatnonzero(months <= 59)
    valid_indices = np.flatnonzero((months >= 62) & (months <= 70) & (months != 66))
    arrays = open_arrays()
    static = np.load(STATIC, mmap_mode="r")
    norm = load_bundled_norm()
    stats = static_norm(static, train_indices)
    target_scale = float(np.std(target[train_indices]))
    train_ds = SequenceDataset(arrays, static, train_indices, norm, stats, target, target_scale)
    valid_ds = SequenceDataset(arrays, static, valid_indices, norm, stats, target, target_scale)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, EVAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model = ThreeStreamTSMixer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = FOLD / "recovery_checkpoint.pt"
    best_path = FOLD / "best_model.pt"
    start_epoch, best_score, best_epoch, history = 1, -1e9, 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"])
        best_score, best_epoch, history = state["best_score"], state["best_epoch"], state["history"]
        start_epoch = state["epoch"] + 1
        print(f"RESUME epoch={start_epoch}", flush=True)
    print(f"GPU={torch.cuda.get_device_name(0)} params={sum(p.numel() for p in model.parameters()):,} train={len(train_indices):,} valid={len(valid_indices):,}", flush=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.time()
        loss = train_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        prediction = predict(model, valid_loader, device, target_scale)
        score = cosine(np.asarray(target[valid_indices]), prediction)
        history.append({"epoch": epoch, "loss": loss, "cosine": score, "seconds": time.time() - started})
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "score": score, "epoch": epoch}, best_path)
            np.save(FOLD / "validation_predictions.npy", prediction)
        recovery = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "generator": generator.get_state(), "epoch": epoch,
            "best_score": best_score, "best_epoch": best_epoch, "history": history,
        }
        torch.save(recovery, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        (FOLD / "score_only.json").write_text(json.dumps({"state": "running", "epoch": epoch, "best_cosine": best_score}, indent=2))
        print(json.dumps(history[-1]), flush=True)
    prediction = np.load(FOLD / "validation_predictions.npy")
    truth = np.asarray(target[valid_indices])
    valid_months = np.asarray(months[valid_indices])
    validation = pd.DataFrame({"sample_id": np.asarray(ids[valid_indices]), "month": valid_months, "target": truth, "prediction": prediction})
    validation.to_feather(FOLD / "validation_predictions.feather")
    fusion = fusion_analysis(validation.sample_id.to_numpy(), valid_months, truth, prediction)
    summary = {
        "experiment": "EXP-SEQUENCE-021-TSMIXER-DEV", "status": "complete",
        "architecture": "patch summaries (mean/std/last), 3 TSMixer blocks per stream, static379 fourth source",
        "train_months": "0-59", "purge_months": [60, 61], "validation_months": "62-70 excluding66",
        "best_epoch": best_epoch, "best_cosine": best_score, "baseline_transformer": 0.15538451490216448,
        "delta_vs_transformer": best_score - 0.15538451490216448, "history": history,
        "fusion_test": fusion,
    }
    for path in (RUN / "score_only.json", RUN / "result.json"):
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

