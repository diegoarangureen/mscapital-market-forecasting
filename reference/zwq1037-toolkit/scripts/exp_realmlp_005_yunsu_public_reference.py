"""Yunsu public-feature RQ RealMLP on the two fixed forward validation folds.

This is the reference experiment for later feature replacement.  The model,
loss, optimiser, target rounding, label noise, EMA, and 16-member ensemble
follow the public notebook.  Preprocessing is fitted inside each training fold
so the local validation score remains usable for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_CSV = PROJECT / "data/interim/public_features/rfmf_0726data/train.csv"
CACHE = PROJECT / "data/interim/public_features/rfmf_0726data/numpy_cache"
MONTH_CACHE = PROJECT / "data/interim/kaggle_relative319_dev"
RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-005-YUNSU-PUBLIC-RQ-REFERENCE"

FOLDS = [
    ("train049_valid5059", 49, 50, 59),
    ("train059_valid6270_ex66", 59, 62, 70),
]
N_ROWS = 1_257_637
N_ENSEMBLE = 16
EMBED_DIM = 6
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 2048
EPOCHS = 10
SEED = 2026
ONE_HOT_MAX = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 竞赛指标使用去均值后的余弦相似度。
    # The competition metric is centered cosine similarity.
    true_centered = y_true.astype(np.float64) - float(np.mean(y_true))
    pred_centered = y_pred.astype(np.float64) - float(np.mean(y_pred))
    denominator = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    return float(np.dot(true_centered, pred_centered) / (denominator + 1e-12))


def ensure_numpy_cache() -> tuple[np.memmap, np.memmap, list[str]]:
    """Stream the 2.6 GB CSV once into an ID-indexed float32 memmap."""
    CACHE.mkdir(parents=True, exist_ok=True)
    feature_path = CACHE / "features.npy"
    target_path = CACHE / "targets.npy"
    names_path = CACHE / "feature_columns.json"
    ready_path = CACHE / "READY.json"

    if ready_path.exists() and feature_path.exists() and target_path.exists() and names_path.exists():
        names = json.loads(names_path.read_text(encoding="utf-8"))
        features = np.load(feature_path, mmap_mode="r")
        targets = np.load(target_path, mmap_mode="r")
        if features.shape == (N_ROWS, len(names)) and targets.shape == (N_ROWS,):
            print(f"CACHE ready rows={features.shape[0]} features={features.shape[1]}", flush=True)
            return features, targets, names

    if not PUBLIC_CSV.exists() or PUBLIC_CSV.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Missing complete public feature file: {PUBLIC_CSV}")

    reader = pacsv.open_csv(
        PUBLIC_CSV,
        read_options=pacsv.ReadOptions(block_size=32 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            column_types={"sample_id": pa.int64(), "target": pa.float64()},
        ),
    )
    columns = reader.schema.names
    feature_names = [name for name in columns if name not in {"sample_id", "target"}]
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(N_ROWS, len(feature_names))
    )
    targets = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(N_ROWS,)
    )
    seen = np.zeros(N_ROWS, dtype=np.bool_)
    processed = 0
    started = time.time()
    for batch in reader:
        sample_ids = batch.column("sample_id").to_numpy(zero_copy_only=False).astype(np.int64)
        values = np.column_stack(
            [batch.column(name).to_numpy(zero_copy_only=False) for name in feature_names]
        ).astype(np.float32, copy=False)
        batch_target = batch.column("target").to_numpy(zero_copy_only=False).astype(np.float32)
        if sample_ids.min() < 0 or sample_ids.max() >= N_ROWS:
            raise AssertionError("Public sample_id is outside the expected dense range")
        features[sample_ids] = values
        targets[sample_ids] = batch_target
        seen[sample_ids] = True
        processed += len(sample_ids)
        if processed // 100_000 != (processed - len(sample_ids)) // 100_000:
            print(f"CACHE rows={processed}/{N_ROWS} elapsed={time.time() - started:.1f}s", flush=True)
    if processed != N_ROWS or not bool(seen.all()):
        raise AssertionError(f"Incomplete cache: processed={processed}, all_seen={seen.all()}")
    features.flush()
    targets.flush()
    names_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    ready_path.write_text(
        json.dumps({"rows": N_ROWS, "features": len(feature_names)}, indent=2), encoding="utf-8"
    )
    print(f"CACHE completed features={len(feature_names)} elapsed={time.time() - started:.1f}s", flush=True)
    return np.load(feature_path, mmap_mode="r"), np.load(target_path, mmap_mode="r"), feature_names


def absolute_correlations(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute feature-target and feature-feature Pearson correlations."""
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean()
    x_norm = np.sqrt(np.sum(x * x, axis=0))
    y_norm = float(np.sqrt(np.sum(y * y)))
    target_corr = np.abs((x.T @ y) / (x_norm * y_norm + 1e-30))
    corr = np.abs((x.T @ x) / (x_norm[:, None] * x_norm[None, :] + 1e-30))
    return target_corr, corr


def select_features(
    train_ids: np.ndarray,
    train_features: np.ndarray,
    train_target: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict]:
    """Reproduce Yunsu's correlation pruning, fitted on the fold's train rows."""
    # 原 notebook 把 sample_id 也放进相关性筛选；这里保留该行为以复刻结构。
    # The public notebook includes sample_id in correlation pruning, so the reference does too.
    correlation_input = np.column_stack((train_ids.astype(np.float64), train_features))
    all_names = ["sample_id", *feature_names]
    target_corr, corr = absolute_correlations(correlation_input, train_target)
    pairs = []
    upper_i, upper_j = np.triu_indices(corr.shape[0], k=1)
    pair_values = corr[upper_i, upper_j]
    high = np.flatnonzero(pair_values >= 0.9)
    order = high[np.argsort(pair_values[high])[::-1]]
    dropped: set[int] = set()
    for position in order:
        left = int(upper_i[position])
        right = int(upper_j[position])
        if left in dropped or right in dropped:
            continue
        dropped.add(right if target_corr[left] >= target_corr[right] else left)
        pairs.append((left, right, float(pair_values[position])))
    standard_deviation = np.std(correlation_input, axis=0)
    for index in range(correlation_input.shape[1]):
        if standard_deviation[index] == 0.0 or target_corr[index] < 0.0001:
            dropped.add(index)

    keep_feature_indices = np.asarray(
        [index - 1 for index in range(1, len(all_names)) if index not in dropped], dtype=np.int64
    )
    kept_names = [feature_names[index] for index in keep_feature_indices]
    details = {
        "input_features": len(feature_names),
        "kept_features": len(kept_names),
        "dropped_features": [all_names[index] for index in sorted(dropped) if index > 0],
        "sample_id_dropped": 0 in dropped,
        "high_correlation_pairs_processed": len(pairs),
    }
    return keep_feature_indices, kept_names, details


def cut_with_train_quantiles(
    train_values: np.ndarray, valid_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the public notebook's 40-bin quantile transformation."""
    quantiles = np.linspace(0.0, 1.0, 41)
    bins = np.unique(np.quantile(train_values, quantiles))
    if len(bins) < 2:
        return train_values.astype(np.float32), valid_values.astype(np.float32)

    def encode(values: np.ndarray, replace_outside: bool) -> np.ndarray:
        encoded = np.searchsorted(bins, values, side="left") - 1
        encoded[values == bins[0]] = 0
        outside = (~np.isfinite(values)) | (values < bins[0]) | (values > bins[-1])
        encoded = np.clip(encoded, 0, len(bins) - 2).astype(np.float32)
        if replace_outside:
            encoded[outside] = 20.0
        return encoded

    return encode(train_values, False), encode(valid_values, True)


def prepare_fold_features(
    train_raw: np.ndarray, valid_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    """Fit public binning, categorical mapping, and robust scaling on one train fold."""
    train = np.asarray(train_raw, dtype=np.float32).copy()
    valid = np.asarray(valid_raw, dtype=np.float32).copy()
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise AssertionError("The public feature matrix unexpectedly contains non-finite values")

    unique_counts = []
    for column in range(train.shape[1]):
        unique_count = int(np.unique(train[:, column]).size)
        if unique_count > 100:
            train[:, column], valid[:, column] = cut_with_train_quantiles(
                train[:, column], valid[:, column]
            )
            unique_count = int(np.unique(train[:, column]).size)
        unique_counts.append(unique_count)

    categorical_indices = [index for index, count in enumerate(unique_counts) if count <= ONE_HOT_MAX]
    numerical_indices = [index for index, count in enumerate(unique_counts) if count > ONE_HOT_MAX]
    cat_dims = []
    train_cat = np.empty((len(train), len(categorical_indices)), dtype=np.float32)
    valid_cat = np.empty((len(valid), len(categorical_indices)), dtype=np.float32)
    for output_column, source_column in enumerate(categorical_indices):
        # 映射顺序跟随训练数据首次出现的顺序，与公开 notebook 一致。
        # Category IDs follow first appearance in the training data, as in the public notebook.
        categories, first_indices = np.unique(train[:, source_column], return_index=True)
        categories = categories[np.argsort(first_indices)]
        mapping = {float(value): index for index, value in enumerate(categories)}
        train_cat[:, output_column] = np.fromiter(
            (mapping[float(value)] for value in train[:, source_column]),
            dtype=np.float32,
            count=len(train),
        )
        valid_cat[:, output_column] = np.fromiter(
            (mapping.get(float(value), 0) for value in valid[:, source_column]),
            dtype=np.float32,
            count=len(valid),
        )
        cat_dims.append(len(categories))

    train_num = train[:, numerical_indices].astype(np.float32, copy=True)
    valid_num = valid[:, numerical_indices].astype(np.float32, copy=True)
    median = np.median(train_num, axis=0)
    q25 = np.quantile(train_num, 0.25, axis=0)
    q75 = np.quantile(train_num, 0.75, axis=0)
    quantile_difference = q75 - q25
    zero = quantile_difference == 0.0
    if np.any(zero):
        value_range = np.max(train_num, axis=0) - np.min(train_num, axis=0)
        quantile_difference[zero] = 0.5 * value_range[zero]
    factors = 1.0 / (quantile_difference + 1e-30)
    factors[quantile_difference == 0.0] = 0.0

    def robust_transform(values: np.ndarray) -> np.ndarray:
        scaled = factors[None, :] * (values - median[None, :])
        return (scaled / np.sqrt(1.0 + (scaled / 3.0) ** 2)).astype(np.float32)

    metadata = {
        "categorical_features": len(categorical_indices),
        "numerical_features": len(numerical_indices),
        "cat_dims": cat_dims,
    }
    return robust_transform(train_num), robust_transform(valid_num), train_cat, valid_cat, cat_dims, metadata


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(N_ENSEMBLE, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None]


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, cat_dims: list[int]):
        super().__init__()
        self.cat_dims = cat_dims
        self.onehot_features = [i for i, dim in enumerate(cat_dims) if dim <= ONE_HOT_MAX]
        self.embed_features = [i for i, dim in enumerate(cat_dims) if dim > ONE_HOT_MAX]
        self.embed_dims = [cat_dims[i] for i in self.embed_features]
        self.embed_offsets = np.cumsum([0, *self.embed_dims[:-1]]).tolist()
        self.per_ensemble_offset = sum(self.embed_dims)
        if self.embed_features:
            self.combined_embedding = nn.Embedding(
                self.per_ensemble_offset * N_ENSEMBLE, EMBED_DIM, padding_idx=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []
        if self.onehot_features:
            values = x[:, :, self.onehot_features].long()
            dimensions = [self.cat_dims[i] for i in self.onehot_features]
            onehot = torch.zeros(
                batch_size, N_ENSEMBLE, sum(dimensions), device=x.device, dtype=torch.float32
            )
            start = 0
            for index, dimension in enumerate(dimensions):
                onehot.scatter_(2, values[:, :, index : index + 1] + start, 1.0)
                start += dimension
            outputs.append(onehot)
        if self.embed_features:
            values = x[:, :, self.embed_features].long()
            ensemble_offset = (
                torch.arange(N_ENSEMBLE, device=x.device) * self.per_ensemble_offset
            )
            feature_offset = torch.tensor(self.embed_offsets, device=x.device)
            indices = values + feature_offset[None, None] + ensemble_offset[None, :, None]
            embedded = self.combined_embedding(indices)
            outputs.append(embedded.reshape(batch_size, N_ENSEMBLE, -1))
        if not outputs:
            return torch.empty(batch_size, N_ENSEMBLE, 0, device=x.device)
        return torch.cat(outputs, dim=2)


class PBLDEmbedding(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.w1 = nn.Parameter(torch.randn(N_ENSEMBLE, n_features, 24))
        self.b1 = nn.Parameter(torch.empty(N_ENSEMBLE, n_features, 24).uniform_(-np.pi, np.pi))
        self.w2 = nn.Parameter(torch.randn(N_ENSEMBLE, n_features, 24, 2) / np.sqrt(24))
        self.b2 = nn.Parameter(torch.randn(N_ENSEMBLE, n_features, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        periodic = torch.cos(2 * np.pi * (x[..., None] * self.w1[None] + self.b1[None]))
        transformed = torch.einsum("bnfh,nfhd->bnfd", periodic, self.w2)
        result = torch.cat((x[..., None], F.gelu(transformed + self.b2[None])), dim=-1)
        return result.reshape(x.shape[0], N_ENSEMBLE, self.n_features * 3)


class NTPLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N_ENSEMBLE, input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(N_ENSEMBLE, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bni,nio->bno", x, self.weight) / np.sqrt(self.weight.shape[1]) + self.bias


class RealMLPRQ(nn.Module):
    def __init__(self, n_numerical: int, cat_dims: list[int]):
        super().__init__()
        self.cate = CategoricalFeatureLayer(cat_dims)
        self.num_embed = PBLDEmbedding(n_numerical)
        categorical_width = sum(dim if dim <= ONE_HOT_MAX else EMBED_DIM for dim in cat_dims)
        total_width = n_numerical * 3 + categorical_width
        self.shared = nn.Sequential(
            nn.LayerNorm(total_width),
            ScalingLayer(total_width),
            NTPLinear(total_width, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            NTPLinear(512, 512),
            nn.GELU(),
            nn.Dropout(0.01),
            NTPLinear(512, 128),
            nn.GELU(),
            nn.Dropout(0.01),
        )
        self.code_heads = nn.ModuleList([NTPLinear(128, 3) for _ in range(2)])
        self.regression_head = NTPLinear(128, 1)
        mask = torch.ones(N_ENSEMBLE, total_width, dtype=torch.bool)
        for member in range(N_ENSEMBLE):
            mask[member, member :: N_ENSEMBLE // 2] = False
        self.register_buffer("feature_mask", mask)

    def forward(
        self, numerical: torch.Tensor, categorical: torch.Tensor, return_codes: bool = False
    ):
        numerical = numerical[:, None].expand(-1, N_ENSEMBLE, -1)
        categorical = categorical[:, None].expand(-1, N_ENSEMBLE, -1)
        combined = torch.cat((self.num_embed(numerical), self.cate(categorical)), dim=2)
        features = self.shared(combined * self.feature_mask[None].float())
        regression = self.regression_head(features)
        if return_codes:
            return [head(features) for head in self.code_heads], regression
        return regression.mean(dim=1)


class RQKMeansEncoder:
    def __init__(self):
        self.codebooks = []

    def fit(self, target: np.ndarray) -> "RQKMeansEncoder":
        residual = np.asarray(target, dtype=np.float64).reshape(-1, 1)
        # 公开 notebook 拟合三层编码，但模型只使用前两个辅助头。
        # The public notebook fits three codebooks while the model uses the first two heads.
        for _ in range(3):
            codebook = KMeans(n_clusters=3, random_state=42, n_init=10).fit(residual)
            codes = codebook.predict(residual)
            residual = residual - codebook.cluster_centers_[codes]
            self.codebooks.append(codebook)
        return self

    def encode(self, target: np.ndarray) -> np.ndarray:
        residual = np.asarray(target, dtype=np.float64).reshape(-1, 1)
        codes = []
        for codebook in self.codebooks:
            layer_codes = codebook.predict(residual)
            codes.append(layer_codes)
            residual = residual - codebook.cluster_centers_[layer_codes]
        return np.stack(codes, axis=1).astype(np.int64)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.model = model
        self.decay = decay
        self.state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self) -> None:
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.state[name].mul_(self.decay).add_(parameter.data, alpha=1.0 - self.decay)

    def apply(self) -> dict[str, torch.Tensor]:
        original = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                original[name] = parameter.data.detach().clone()
                parameter.data.copy_(self.state[name])
        return original

    def restore(self, original: dict[str, torch.Tensor]) -> None:
        for name, parameter in self.model.named_parameters():
            if name in original:
                parameter.data.copy_(original[name])


def flat_anneal(initial_value: float, progress: float, flat_ratio: float = 0.5) -> float:
    if progress < flat_ratio:
        return initial_value
    return initial_value * (1.0 - (progress - flat_ratio) / (1.0 - flat_ratio))


def parameter_groups(model: nn.Module) -> list[dict]:
    scale_parameters = []
    pbld_parameters = []
    first_parameters = []
    other_weights = []
    bias_parameters = []
    for name, parameter in model.named_parameters():
        if "scale" in name:
            scale_parameters.append(parameter)
        elif "num_embed" in name:
            pbld_parameters.append(parameter)
        elif name == "shared.0.weight":
            first_parameters.append(parameter)
        elif "bias" in name:
            bias_parameters.append(parameter)
        else:
            other_weights.append(parameter)
    return [
        {"params": scale_parameters, "lr": LEARNING_RATE * 20.0, "weight_decay": 0.001},
        {"params": pbld_parameters, "lr": LEARNING_RATE * 0.093, "weight_decay": 0.01},
        {"params": first_parameters, "lr": LEARNING_RATE, "weight_decay": 0.001},
        {"params": other_weights, "lr": LEARNING_RATE, "weight_decay": 0.01},
        {"params": bias_parameters, "lr": LEARNING_RATE * 0.1, "weight_decay": 0.005},
    ]


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    code_logits: list[torch.Tensor],
    codes: torch.Tensor,
    rq_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction.squeeze(-1)
    repeated_target = target[:, None].expand_as(prediction)
    prediction_flat = prediction.reshape(-1)
    target_flat = repeated_target.reshape(-1)
    sample_weight = torch.where(target_flat.abs() > 0.001, 0.5, 1.0)
    mse = (sample_weight * (prediction_flat - target_flat).square()).mean()
    pred_centered = prediction_flat - prediction_flat.mean()
    target_centered = target_flat - target_flat.mean()
    cosine = F.cosine_similarity(pred_centered, target_centered, dim=0)
    rq_loss = torch.zeros((), device=prediction.device)
    expanded_codes = codes[:, None].expand(-1, N_ENSEMBLE, -1)
    for layer, logits in enumerate(code_logits):
        labels = expanded_codes[:, :, layer].reshape(-1)
        rq_loss = rq_loss + F.cross_entropy(logits.reshape(-1, 3), labels)
    rq_loss = rq_loss / len(code_logits)
    return mse + 0.01 * (1.0 - cosine) + rq_weight * rq_loss, cosine, mse, rq_loss


@torch.no_grad()
def evaluate(
    model: nn.Module, numerical: torch.Tensor, categorical: torch.Tensor, target: np.ndarray
) -> tuple[float, np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(numerical), EVAL_BATCH_SIZE):
        prediction = model(
            numerical[start : start + EVAL_BATCH_SIZE],
            categorical[start : start + EVAL_BATCH_SIZE],
        ).squeeze(-1)
        predictions.append(prediction.cpu().numpy())
    result = np.concatenate(predictions)
    return cosine_score(target, result), result


def run_fold(
    fold_name: str,
    train_end: int,
    valid_start: int,
    valid_end: int,
    features: np.memmap,
    target: np.memmap,
    feature_names: list[str],
    sample_ids: np.ndarray,
    months: np.ndarray,
    epochs: int,
    smoke: bool,
) -> dict:
    set_seed(SEED)
    train_indices = np.flatnonzero(months <= train_end)
    valid_mask = (months >= valid_start) & (months <= valid_end)
    if valid_start == 62:
        valid_mask &= months != 66
    valid_indices = np.flatnonzero(valid_mask)
    if smoke:
        train_indices = train_indices[:4096]
        valid_indices = valid_indices[:2048]
        epochs = 1
    print(
        f"FOLD {fold_name} train={len(train_indices)} valid={len(valid_indices)} epochs={epochs}",
        flush=True,
    )

    raw_train = np.asarray(features[train_indices], dtype=np.float32)
    raw_valid = np.asarray(features[valid_indices], dtype=np.float32)
    train_target_raw = np.asarray(target[train_indices], dtype=np.float32)
    valid_target = np.asarray(target[valid_indices], dtype=np.float32)
    # 公开特征含缺失值，而公开训练 notebook 没展示对应填补代码。
    # Public features contain missing values, but the public training notebook omits imputation.
    # Fit a median on this fold's training rows so both the reference and later replacements
    # use the same leakage-safe, minimal completion of the published pipeline.
    raw_train = raw_train.copy()
    raw_valid = raw_valid.copy()
    raw_train[~np.isfinite(raw_train)] = np.nan
    raw_valid[~np.isfinite(raw_valid)] = np.nan
    fill_values = np.nanmedian(raw_train, axis=0)
    fill_values = np.nan_to_num(fill_values, nan=0.0).astype(np.float32)
    train_missing = np.where(~np.isfinite(raw_train))
    valid_missing = np.where(~np.isfinite(raw_valid))
    raw_train[train_missing] = fill_values[train_missing[1]]
    raw_valid[valid_missing] = fill_values[valid_missing[1]]
    kept_indices, kept_names, selection = select_features(
        sample_ids[train_indices], raw_train, train_target_raw, feature_names
    )
    raw_train = raw_train[:, kept_indices]
    raw_valid = raw_valid[:, kept_indices]
    print(
        f"FOLD {fold_name} selected={len(kept_names)}/{len(feature_names)} "
        f"sample_id_dropped={selection['sample_id_dropped']}",
        flush=True,
    )
    train_num, valid_num, train_cat, valid_cat, cat_dims, preprocessing = prepare_fold_features(
        raw_train, raw_valid
    )
    del raw_train, raw_valid
    train_target = np.round(train_target_raw, 4).astype(np.float32)
    encoder = RQKMeansEncoder().fit(train_target)
    train_codes = encoder.encode(train_target)

    device = torch.device("cuda")
    train_num_gpu = torch.from_numpy(train_num).to(device)
    valid_num_gpu = torch.from_numpy(valid_num).to(device)
    train_cat_gpu = torch.from_numpy(train_cat).to(device)
    valid_cat_gpu = torch.from_numpy(valid_cat).to(device)
    train_target_gpu = torch.from_numpy(train_target).to(device)
    train_codes_gpu = torch.from_numpy(train_codes).to(device)
    del train_num, valid_num, train_cat, valid_cat, train_codes

    model = RealMLPRQ(train_num_gpu.shape[1], cat_dims).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model), betas=(0.9, 0.98))
    ema = EMA(model, decay=0.998)
    steps_per_epoch = (len(train_indices) + TRAIN_BATCH_SIZE - 1) // TRAIN_BATCH_SIZE
    total_steps = steps_per_epoch * epochs
    best_score = -1.0
    best_epoch = 0
    best_prediction = None
    history = []
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator)
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(permutation), TRAIN_BATCH_SIZE):
            cpu_indices = permutation[start : start + TRAIN_BATCH_SIZE]
            indices = cpu_indices.to(device)
            global_step = (epoch - 1) * steps_per_epoch + start // TRAIN_BATCH_SIZE
            progress = min(global_step / total_steps, 1.0)
            learning_rates = [
                LEARNING_RATE * 20.0,
                LEARNING_RATE * 0.093,
                LEARNING_RATE,
                LEARNING_RATE,
                LEARNING_RATE * 0.1,
            ]
            for group, initial_rate in zip(optimizer.param_groups, learning_rates):
                group["lr"] = flat_anneal(initial_rate, progress)
            noisy_target = train_target_gpu[indices] + torch.randn_like(train_target_gpu[indices]) * (
                0.005 * (1.0 - progress)
            )
            optimizer.zero_grad(set_to_none=True)
            code_logits, prediction = model(
                train_num_gpu[indices], train_cat_gpu[indices], return_codes=True
            )
            loss, train_cosine, mse, rq_loss = compute_loss(
                prediction,
                noisy_target,
                code_logits,
                train_codes_gpu[indices],
                flat_anneal(0.1, progress),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            sums += [float(loss.detach()), float(train_cosine.detach()), float(mse.detach()), float(rq_loss.detach())]
            batches += 1
        original = ema.apply()
        validation_score, validation_prediction = evaluate(
            model, valid_num_gpu, valid_cat_gpu, valid_target
        )
        ema.restore(original)
        row = {
            "epoch": epoch,
            "loss": float(sums[0] / batches),
            "train_cosine": float(sums[1] / batches),
            "mse": float(sums[2] / batches),
            "rq_loss": float(sums[3] / batches),
            "validation_cosine": validation_score,
        }
        history.append(row)
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_prediction = validation_prediction.copy()
        print(
            f"EPOCH {fold_name} {epoch}/{epochs} loss={row['loss']:.6f} "
            f"train_cos={row['train_cosine']:.6f} val={validation_score:.6f} "
            f"best={best_score:.6f}@{best_epoch} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    fold_dir = RUN_DIR / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "validation_sample_ids.npy", sample_ids[valid_indices])
    np.save(fold_dir / "validation_targets.npy", valid_target)
    np.save(fold_dir / "validation_predictions.npy", best_prediction)
    result = {
        "fold": fold_name,
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "best_epoch": best_epoch,
        "best_validation_cosine": best_score,
        "kept_feature_names": kept_names,
        "selection": selection,
        "preprocessing": preprocessing,
        "history": history,
    }
    (fold_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del train_num_gpu, valid_num_gpu, train_cat_gpu, valid_cat_gpu
    del train_target_gpu, train_codes_gpu, model, optimizer, ema
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fold", choices=["both", "first", "late"], default="both")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    features, target, feature_names = ensure_numpy_cache()
    sample_ids = np.load(MONTH_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(MONTH_CACHE / "months.npy", mmap_mode="r")
    reference_target = np.load(MONTH_CACHE / "targets.npy", mmap_mode="r")
    if not np.array_equal(sample_ids, np.arange(N_ROWS)):
        raise AssertionError("Expected dense sample IDs 0..N-1")
    maximum_target_difference = float(np.max(np.abs(target - reference_target)))
    print(
        f"DATA rows={len(sample_ids)} public_features={len(feature_names)} "
        f"max_target_difference={maximum_target_difference:.3g}",
        flush=True,
    )
    if maximum_target_difference > 1e-7:
        raise AssertionError("Public target does not align with the competition label cache")

    folds = FOLDS
    if args.fold == "first":
        folds = FOLDS[:1]
    elif args.fold == "late":
        folds = FOLDS[1:]
    results = [
        run_fold(
            *fold,
            features,
            target,
            feature_names,
            sample_ids,
            months,
            args.epochs,
            args.smoke,
        )
        for fold in folds
    ]
    summary = {
        "experiment": "EXP-REALMLP-005-YUNSU-PUBLIC-RQ-REFERENCE",
        "model": "Yunsu public RQ RealMLP",
        "public_feature_count": len(feature_names),
        "n_ensemble": N_ENSEMBLE,
        "epochs": 1 if args.smoke else args.epochs,
        "smoke": args.smoke,
        "folds": [
            {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_validation_cosine": result["best_validation_cosine"],
                "kept_features": result["selection"]["kept_features"],
            }
            for result in results
        ],
    }
    output_name = "smoke_summary.json" if args.smoke else "score_summary.json"
    (RUN_DIR / output_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

