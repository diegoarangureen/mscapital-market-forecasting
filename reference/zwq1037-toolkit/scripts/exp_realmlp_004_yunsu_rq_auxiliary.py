"""Low-memory numeric-only pilot of Yunsu's PBLD RealMLP on our 379 features."""

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "data/interim/kaggle_kernels/relative319_xs_tabm_notebook"))
import run_extracted as recipe
from build_order_quote_position_features import FEATURE_COLUMNS

ROOT = PROJECT / "data/interim"
RUN = ROOT / "tree_experiments/EXP-REALMLP-004-YUNSU-RQ-AUXILIARY"
FOLDS = [
    ("train049_valid5059", 49, 50, 59),
    ("train059_valid6270_ex66", 59, 62, 70),
]
MEMBERS = 4
EPOCHS = 4
BATCH_SIZE = 1024
SEED = 42


def cosine(y, p):
    return float(np.dot(y, p) / (np.linalg.norm(y) * np.linalg.norm(p) + 1e-12))


class PBLDEmbedding(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(MEMBERS, n_features, 24))
        self.b1 = nn.Parameter(torch.empty(MEMBERS, n_features, 24).uniform_(-np.pi, np.pi))
        self.w2 = nn.Parameter(torch.randn(MEMBERS, n_features, 24, 2) / np.sqrt(24))
        self.b2 = nn.Parameter(torch.randn(MEMBERS, n_features, 2))

    def forward(self, x):
        # 原始数值和周期变换共同进入网络。
        # Keep each raw value beside its periodic learned representation.
        periodic = torch.cos(2 * np.pi * (x[..., None] * self.w1[None] + self.b1[None]))
        transformed = torch.einsum("bnfh,nfhd->bnfd", periodic, self.w2)
        result = torch.cat((x[..., None], F.gelu(transformed + self.b2[None])), dim=-1)
        return result.flatten(start_dim=2)


class NTPLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(MEMBERS, input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(MEMBERS, output_dim))

    def forward(self, x):
        return torch.einsum("bni,nio->bno", x, self.weight) / np.sqrt(self.weight.shape[1]) + self.bias


class RQKMeansEncoder:
    def __init__(self, n_layers=2, codebook_size=3):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.codebooks = []

    def fit(self, target):
        residual = np.asarray(target, dtype=np.float64).reshape(-1, 1)
        for _ in range(self.n_layers):
            codebook = KMeans(
                n_clusters=self.codebook_size,
                random_state=SEED,
                n_init=10,
            ).fit(residual)
            codes = codebook.predict(residual)
            residual = residual - codebook.cluster_centers_[codes]
            self.codebooks.append(codebook)
        return self

    def encode(self, target):
        residual = np.asarray(target, dtype=np.float64).reshape(-1, 1)
        result = []
        for codebook in self.codebooks:
            codes = codebook.predict(residual)
            result.append(codes)
            residual = residual - codebook.cluster_centers_[codes]
        return np.stack(result, axis=1).astype(np.int64)


def flat_anneal(initial_value, progress, flat_ratio=0.5):
    if progress < flat_ratio:
        return initial_value
    decay_progress = (progress - flat_ratio) / (1.0 - flat_ratio)
    return initial_value * (1.0 - decay_progress)

class NumericYunsuMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        width = n_features * 3
        self.embed = PBLDEmbedding(n_features)
        self.norm = nn.LayerNorm(width)
        self.scale = nn.Parameter(torch.ones(MEMBERS, width))
        self.layers = nn.ModuleList([
            NTPLinear(width, 512),
            NTPLinear(512, 512),
            NTPLinear(512, 128),
        ])
        self.reg_head = NTPLinear(128, 1)
        self.code_heads = nn.ModuleList([NTPLinear(128, 3) for _ in range(2)])
        mask = torch.ones(MEMBERS, width)
        for member in range(MEMBERS):
            # 四成员试跑维持原模型约八分之一的特征遮盖比例。
            # Preserve the original approximate mask fraction in this four-member pilot.
            mask[member, member::8] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        x = x[:, None, :].expand(-1, MEMBERS, -1)
        x = self.embed(x) * self.mask[None]
        x = self.norm(x) * self.scale[None]
        for layer in self.layers:
            x = F.dropout(F.gelu(layer(x)), p=0.01, training=self.training)
        member_prediction = self.reg_head(x).squeeze(-1)
        code_logits = [head(x) for head in self.code_heads]
        return member_prediction, code_logits


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    RUN.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "kaggle_relative319_dev"
    base = np.load(cache / "features.npy", mmap_mode="r")
    ids = np.load(cache / "sample_ids.npy", mmap_mode="r")
    months = np.load(cache / "months.npy", mmap_mode="r")
    target = np.load(cache / "targets.npy", mmap_mode="r")
    columns = json.loads((cache / "feature_columns.json").read_text(encoding="utf-8"))
    if base.shape[1] != 319:
        raise AssertionError(base.shape)
    xs40, names = recipe.add_relative_features(base, months, columns)
    state = pd.read_feather(PROJECT / "data/processed/train_order_quote_position_features.feather")
    state = state.sort_values("sample_id")
    if not np.array_equal(state.sample_id.to_numpy(), ids):
        raise AssertionError("Order feature IDs differ from cache")
    order20 = state[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    del state
    if len(names) != 40 or order20.shape[1] != 20:
        raise AssertionError("Expected 379 features")

    def raw_rows(indices):
        return np.concatenate((base[indices], xs40[indices], order20[indices]), axis=1)

    summaries = []
    for fold_name, train_end, valid_start, valid_end in FOLDS:
        torch.manual_seed(SEED)
        train_idx = np.flatnonzero(months <= train_end)
        valid_idx = np.flatnonzero((months >= valid_start) & (months <= valid_end))
        if valid_start == 62:
            valid_idx = valid_idx[months[valid_idx] != 66]
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(train_idx, size=min(50000, len(train_idx)), replace=False)
        sample = raw_rows(sample_idx)
        sample[~np.isfinite(sample)] = np.nan
        median = np.nanmedian(sample, axis=0)
        q25 = np.nanpercentile(sample, 25, axis=0)
        q75 = np.nanpercentile(sample, 75, axis=0)
        median = np.nan_to_num(median, nan=0).astype(np.float32)
        scale = np.maximum(np.nan_to_num(q75 - q25, nan=1), 1e-4).astype(np.float32)
        del sample
        target_mean = float(target[train_idx].mean())
        target_std = float(target[train_idx].std())
        scaled_train_target = (
            (np.asarray(target[train_idx], dtype=np.float64) - target_mean) / target_std
        )
        rq_encoder = RQKMeansEncoder(n_layers=2, codebook_size=3).fit(
            scaled_train_target
        )
        rq_codes = np.zeros((len(target), 2), dtype=np.int64)
        rq_codes[train_idx] = rq_encoder.encode(scaled_train_target)

        # 分块完成一次 CPU 预处理，然后用 float16 常驻显存。
        # Preprocess once in chunks, then keep float16 features on the GPU.
        gpu_features = torch.empty((len(ids), 379), dtype=torch.float16, device="cuda")
        prepare_rows = 16384
        for start in range(0, len(ids), prepare_rows):
            stop = min(start + prepare_rows, len(ids))
            x = raw_rows(slice(start, stop))
            x = np.where(np.isfinite(x), x, median)
            x = np.clip((x - median) / scale, -10, 10).astype(np.float16)
            gpu_features[start:stop].copy_(torch.from_numpy(x), non_blocking=False)
        gpu_target = torch.from_numpy(np.asarray(target, dtype=np.float32).copy()).to("cuda")
        gpu_rq_codes = torch.from_numpy(rq_codes).to("cuda")
        model = NumericYunsuMLP(379).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        baseline_file = ROOT / "tree_experiments/EXP-TABM-026-ORDER-QUOTE-POSITION20" / fold_name / "validation_predictions.feather"
        baseline = pd.read_feather(baseline_file).sort_values("sample_id")
        if valid_start == 62:
            baseline = baseline[baseline.month != 66]
        if not np.array_equal(baseline.sample_id.to_numpy(), ids[valid_idx]):
            raise AssertionError("Baseline validation IDs differ")
        baseline_pred = baseline.candidate.to_numpy(dtype=np.float64)
        y_valid = np.asarray(target[valid_idx], dtype=np.float64)
        best = None
        steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
        total_steps = steps_per_epoch * EPOCHS
        for epoch in range(1, EPOCHS + 1):
            model.train()
            shuffled = rng.permutation(train_idx)
            losses = []
            rq_losses = []
            for start in range(0, len(shuffled), BATCH_SIZE):
                batch_idx = shuffled[start:start + BATCH_SIZE]
                gpu_idx = torch.as_tensor(batch_idx, dtype=torch.long, device="cuda")
                x = gpu_features[gpu_idx]
                y = (gpu_target[gpu_idx] - target_mean) / target_std
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    member_prediction, code_logits = model(x)
                    member_prediction = member_prediction.float()
                    repeated_target = y[:, None].expand_as(member_prediction)
                    raw_target = gpu_target[gpu_idx]
                    sample_weight = torch.where(raw_target.abs() > 0.001, 0.5, 1.0)[:, None]
                    mse_loss = (sample_weight * (member_prediction - repeated_target).square()).mean()
                    pred_flat = member_prediction.flatten()
                    target_flat = repeated_target.flatten()
                    cosine_loss = 1.0 - F.cosine_similarity(
                        pred_flat - pred_flat.mean(), target_flat - target_flat.mean(), dim=0
                    )
                    batch_codes = gpu_rq_codes[gpu_idx]
                    rq_loss = torch.zeros((), dtype=torch.float32, device="cuda")
                    for layer_index, logits in enumerate(code_logits):
                        labels = batch_codes[:, layer_index, None].expand(
                            -1, MEMBERS
                        ).reshape(-1)
                        rq_loss = rq_loss + F.cross_entropy(
                            logits.float().reshape(-1, 3), labels
                        )
                    rq_loss = rq_loss / len(code_logits)
                    global_step = (
                        (epoch - 1) * steps_per_epoch + start // BATCH_SIZE
                    )
                    progress = min(global_step / total_steps, 1.0)
                    rq_weight = flat_anneal(0.1, progress)
                    loss = mse_loss + 0.01 * cosine_loss + rq_weight * rq_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                rq_losses.append(float(rq_loss.detach()))
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(valid_idx), BATCH_SIZE):
                    gpu_idx = torch.as_tensor(valid_idx[start:start + BATCH_SIZE], dtype=torch.long, device="cuda")
                    with torch.autocast("cuda", dtype=torch.float16):
                        member_output, _ = model(gpu_features[gpu_idx])
                        chunks.append(member_output.mean(dim=1).float().cpu().numpy())
            pred = np.concatenate(chunks).astype(np.float64) * target_std + target_mean
            standalone = cosine(y_valid, pred)
            blend = cosine(y_valid, 0.8 * baseline_pred + 0.2 * pred)
            row = {"fold": fold_name, "epoch": epoch, "train_loss": float(np.mean(losses)), "train_rq_loss": float(np.mean(rq_losses)),
                   "standalone": standalone, "tabm_baseline": cosine(y_valid, baseline_pred),
                   "tabm80_mlp20": blend, "blend_delta": blend - cosine(y_valid, baseline_pred)}
            print(json.dumps(row), flush=True)
            if best is None or blend > best["tabm80_mlp20"]:
                best = row
                pd.DataFrame({"sample_id": ids[valid_idx], "month": months[valid_idx],
                              "target": y_valid, "mlp": pred, "tabm": baseline_pred}
                             ).to_feather(RUN / f"{fold_name}_validation_predictions.feather")
            (RUN / "score_summary.json").write_text(json.dumps({"epochs": summaries + [best]}, indent=2), encoding="utf-8")
        summaries.append(best)
        del model, optimizer, scaler, gpu_features, gpu_target, gpu_rq_codes, rq_codes
        torch.cuda.empty_cache()
        # 只有首段出现增益，才投入计算较晚的第二段。
        # Advance to the later window only when the first screen is positive.
        if best["blend_delta"] <= 0:
            break
    (RUN / "score_summary.json").write_text(json.dumps({"best_by_fold": summaries, "pilot": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()




