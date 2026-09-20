"""Prepare strict prefix-prediction pretraining for the proven factorized Transformer."""
import ast
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
KERNEL = PROJECT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev"
SOURCE = KERNEL / "run_v7_baseline.py"


PRETRAIN_CODE = r'''
PRETRAIN_MARKET_BINS = 5
PRETRAIN_FLOW_BINS = 15
PRETRAIN_EVAL_START_MONTH = 54
PRETRAIN_EPOCHS = 1


def build_prefix_pretrain_targets(arrays, months, outer_train_indices, chunk_rows=8192):
    """Use real quotes on both sides of the hidden final 15-second interval."""
    n_rows = len(months)
    count_path = feature_paths("train")["market_count"]
    market_count = np.memmap(
        count_path, dtype=np.float16, mode="r", shape=(n_rows, MARKET_LEN)
    )
    targets = np.full(n_rows, np.nan, dtype=np.float32)
    eligible_parts = []
    outer_train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    hidden_start = MARKET_LEN - PRETRAIN_MARKET_BINS
    for start in range(0, outer_train_indices.size, chunk_rows):
        rows = outer_train_indices[start:start + chunk_rows]
        counts = np.asarray(market_count[rows], dtype=np.float32)
        visible_mask = counts[:, :hidden_start] > 0
        hidden_mask = counts[:, hidden_start:] > 0
        valid = visible_mask.any(axis=1) & hidden_mask.any(axis=1)
        if not valid.any():
            continue
        good_rows = rows[valid]
        visible = visible_mask[valid]
        hidden = hidden_mask[valid]
        visible_last = hidden_start - 1 - np.argmax(visible[:, ::-1], axis=1)
        hidden_last = hidden_start + PRETRAIN_MARKET_BINS - 1 - np.argmax(
            hidden[:, ::-1], axis=1
        )
        raw_market = arrays["market"]
        start_mid = np.asarray(
            raw_market[good_rows, visible_last, 0], dtype=np.float32
        ) + 1.0
        end_mid = np.asarray(
            raw_market[good_rows, hidden_last, 0], dtype=np.float32
        ) + 1.0
        returns = end_mid / np.maximum(start_mid, 1e-6) - 1.0
        finite = np.isfinite(returns) & (start_mid > 0.1) & (end_mid > 0.1)
        finite &= np.abs(returns) <= 0.10
        if finite.any():
            keep_rows = good_rows[finite]
            targets[keep_rows] = returns[finite]
            eligible_parts.append(keep_rows)
    eligible = np.concatenate(eligible_parts) if eligible_parts else np.empty(0, dtype=np.int64)
    train_indices = eligible[months[eligible] < PRETRAIN_EVAL_START_MONTH]
    eval_indices = eligible[
        (months[eligible] >= PRETRAIN_EVAL_START_MONTH)
        & (months[eligible] <= TRAIN_END_MONTH)
    ]
    if train_indices.size == 0 or eval_indices.size == 0:
        raise RuntimeError("No eligible prefix-pretraining rows were found")
    return targets, train_indices, eval_indices


class PrefixPretrainDataset(Dataset):
    def __init__(self, arrays, indices, norm, targets, target_scale):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm = norm
        self.targets = targets
        self.target_scale = target_scale

    def __len__(self):
        return self.indices.size

    def _norm_sequence(self, name, values):
        pad = np.abs(values).sum(axis=-1) == 0
        values = (values - self.norm[name]["mean"]) / self.norm[name]["std"]
        values = np.clip(values, -8.0, 8.0)
        values = np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0)
        values[pad] = 0.0
        return values.astype(np.float32, copy=False)

    def __getitem__(self, item):
        index = self.indices[item]
        market = self._norm_sequence(
            "market", np.asarray(self.arrays["market"][index], dtype=np.float32)
        )
        transaction = self._norm_sequence(
            "tx", np.asarray(self.arrays["tx"][index], dtype=np.float32)
        )
        order = self._norm_sequence(
            "order", np.asarray(self.arrays["order"][index], dtype=np.float32)
        )
        market[-PRETRAIN_MARKET_BINS:] = 0.0
        transaction[-PRETRAIN_FLOW_BINS:] = 0.0
        order[-PRETRAIN_FLOW_BINS:] = 0.0
        scaled_target = np.float32(self.targets[index] / self.target_scale)
        return (
            torch.from_numpy(market),
            torch.from_numpy(transaction),
            torch.from_numpy(order),
            torch.tensor(scaled_target),
        )


def train_pretrain_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_squared_error = 0.0
    row_count = 0
    for market, transaction, order, target in loader:
        market = market.to(device, non_blocking=True).contiguous()
        transaction = transaction.to(device, non_blocking=True).contiguous()
        order = order.to(device, non_blocking=True).contiguous()
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            prediction = model(
                market, transaction, order, static=None, pretrain=True
            )
            prediction = torch.nan_to_num(
                prediction, nan=0.0, posinf=0.0, neginf=0.0
            )
            loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = target.numel()
        total_squared_error += float(loss.detach().cpu()) * count
        row_count += count
    return math.sqrt(total_squared_error / max(row_count, 1))


@torch.no_grad()
def evaluate_pretrain(model, loader, device, target_scale):
    model.eval()
    predictions = []
    targets = []
    for market, transaction, order, target in loader:
        market = market.to(device, non_blocking=True).contiguous()
        transaction = transaction.to(device, non_blocking=True).contiguous()
        order = order.to(device, non_blocking=True).contiguous()
        prediction = model(
            market, transaction, order, static=None, pretrain=True
        )
        predictions.append(prediction.detach().float().cpu().numpy())
        targets.append(target.numpy())
    prediction = np.concatenate(predictions) * target_scale
    target = np.concatenate(targets) * target_scale
    model_rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(target ** 2)))
    return model_rmse, zero_rmse
'''


ARCHITECTURE = r'''
class _FactorizedStreamEncoder(nn.Module):
    def __init__(self, input_dim, length, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, length, d_model))
        self.conv5 = ConvBlock(d_model, kernel=5, dropout=dropout)
        self.conv3 = ConvBlock(d_model, kernel=3, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values):
        values = values.contiguous()
        padding_mask = values.abs().sum(-1) == 0
        safe_mask = padding_mask.clone()
        safe_mask[padding_mask.all(dim=1), -1] = False
        tokens = self.projection(values) + self.position
        tokens = self.conv5(tokens)
        tokens = self.conv3(tokens)
        tokens = self.encoder(tokens, src_key_padding_mask=safe_mask.contiguous())
        logits = self.attention(tokens).squeeze(-1).masked_fill(padding_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * (~padding_mask).to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bl,bld->bd", weights, tokens)


class _JointMultiStreamStaticModel(nn.Module):
    """Share sequence encoders between prefix pretraining and the full task."""
    def __init__(self, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.market = _FactorizedStreamEncoder(
            len(MARKET_FEATURES), MARKET_LEN, d_model, nhead, nlayers, dropout
        )
        self.transaction = _FactorizedStreamEncoder(
            len(TX_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout
        )
        self.order = _FactorizedStreamEncoder(
            len(ORDER_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192), nn.SiLU(), nn.LayerNorm(192),
            nn.Dropout(dropout), nn.Linear(192, d_model), nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.source_attention = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )
        self.pretrain_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)

    def _pool(self, summaries, source_embedding):
        summaries = self.fusion(summaries + source_embedding)
        weights = torch.softmax(
            self.source_attention(summaries).squeeze(-1), dim=1
        ).unsqueeze(-1)
        return (summaries * weights).sum(dim=1)

    def forward(self, market, transaction, order, static=None, pretrain=False):
        sequence_summaries = torch.stack([
            self.market(market),
            self.transaction(transaction),
            self.order(order),
        ], dim=1)
        if pretrain:
            pooled = self._pool(sequence_summaries, self.source_embedding[:, :3])
            return self.pretrain_head(pooled).squeeze(-1)
        if static is None:
            raise ValueError("static features are required for main-task fine-tuning")
        summaries = torch.cat([
            sequence_summaries,
            self.static_adapter(static).unsqueeze(1),
        ], dim=1)
        pooled = self._pool(summaries, self.source_embedding)
        return self.head(pooled).squeeze(-1)
'''


OLD_MAIN_SETUP = '''    target_scale = float(np.std(target[train_idx]))
    print(f"target_scale={target_scale:.8f}", flush=True)
    train_ds = GridDataset(train_arrays, train_idx, norm, target=target, target_scale=target_scale)
    valid_ds = GridDataset(train_arrays, valid_idx, norm, target=target, target_scale=target_scale)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCnnModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")
'''


NEW_MAIN_SETUP = '''    target_scale = float(np.std(target[train_idx]))
    print(f"target_scale={target_scale:.8f}", flush=True)
    train_ds = GridDataset(train_arrays, train_idx, norm, target=target, target_scale=target_scale)
    valid_ds = GridDataset(train_arrays, valid_idx, norm, target=target, target_scale=target_scale)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCnnModel().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    pretrain_targets, pretrain_train_idx, pretrain_eval_idx = build_prefix_pretrain_targets(
        train_arrays, months, train_idx
    )
    pretrain_scale = float(np.std(pretrain_targets[pretrain_train_idx]))
    if not np.isfinite(pretrain_scale) or pretrain_scale < 1e-8:
        raise RuntimeError(f"Invalid pretraining target scale: {pretrain_scale}")
    print(
        f"pretrain_train_rows={pretrain_train_idx.size:,}, "
        f"pretrain_eval_rows={pretrain_eval_idx.size:,}, "
        f"pretrain_target_scale={pretrain_scale:.8f}",
        flush=True,
    )
    pretrain_train_ds = PrefixPretrainDataset(
        train_arrays, pretrain_train_idx, norm, pretrain_targets, pretrain_scale
    )
    pretrain_eval_ds = PrefixPretrainDataset(
        train_arrays, pretrain_eval_idx, norm, pretrain_targets, pretrain_scale
    )
    pretrain_train_loader = DataLoader(
        pretrain_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    pretrain_eval_loader = DataLoader(
        pretrain_eval_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    for pretrain_epoch in range(1, PRETRAIN_EPOCHS + 1):
        pretrain_train_rmse_scaled = train_pretrain_epoch(
            model, pretrain_train_loader, pretrain_optimizer, scaler, device
        )
        print(
            f"pretrain_epoch={pretrain_epoch} "
            f"train_rmse_scaled={pretrain_train_rmse_scaled:.6f}",
            flush=True,
        )
    pretrain_eval_rmse, pretrain_zero_rmse = evaluate_pretrain(
        model, pretrain_eval_loader, device, pretrain_scale
    )
    pretrain_skill = 1.0 - pretrain_eval_rmse / max(pretrain_zero_rmse, 1e-12)
    print(
        f"pretrain_eval_rmse={pretrain_eval_rmse:.8f}, "
        f"zero_return_rmse={pretrain_zero_rmse:.8f}, "
        f"skill={pretrain_skill:.6f}",
        flush=True,
    )
    if pretrain_eval_rmse >= pretrain_zero_rmse:
        failed_result = {
            "experiment": "factorized_transformer_predictive_pretrain",
            "status": "pretraining_gate_failed",
            "pretrain_eval_rmse": pretrain_eval_rmse,
            "zero_return_rmse": pretrain_zero_rmse,
            "pretrain_skill": pretrain_skill,
            "pretrain_train_rows": int(pretrain_train_idx.size),
            "pretrain_eval_rows": int(pretrain_eval_idx.size),
        }
        with open(WORK_DIR / "score_only.json", "w") as output_file:
            json.dump(failed_result, output_file, indent=2)
        with open(WORK_DIR / "result.json", "w") as output_file:
            json.dump(failed_result, output_file, indent=2)
        raise RuntimeError("Prefix prediction did not beat the zero-return baseline")
    del pretrain_train_loader, pretrain_eval_loader
    del pretrain_train_ds, pretrain_eval_ds, pretrain_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))
'''


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return source.replace(old, new, 1)


def main():
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        'WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working/factorized_transformer"))',
        'WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working/factorized_transformer_predictive_pretrain"))',
    )
    source = source.replace(
        'EPOCHS = int(os.environ.get("EPOCHS", "6"))',
        'EPOCHS = int(os.environ.get("EPOCHS", "4"))',
    )

    insertion_point = source.index("class _JointMultiStreamStaticModel(nn.Module):", source.index("class GridDataset(Dataset):", 900))
    source = source[:insertion_point] + PRETRAIN_CODE + "\n\n" + source[insertion_point:]
    source = replace_once(source, OLD_MAIN_SETUP, NEW_MAIN_SETUP, "main setup")

    old_wrapper = '''    def forward(self, market, transaction, order, static):
        return self.parallel(
            market,
            transaction,
            order,
            static,
        )
'''
    new_wrapper = '''    def forward(self, market, transaction, order, static=None, pretrain=False):
        return self.parallel(
            market,
            transaction,
            order,
            static,
            pretrain=pretrain,
        )
'''
    source = replace_once(source, old_wrapper, new_wrapper, "model wrapper")

    final_start = source.rfind("class _FactorizedStreamEncoder(nn.Module):")
    final_end = source.index('if __name__ == "__main__":', final_start)
    source = source[:final_start] + ARCHITECTURE + "\n\n" + source[final_end:]
    source = source.replace(
        '"experiment": "multistream_factorized_transformer_dev"',
        '"experiment": "factorized_transformer_predictive_pretrain"',
    )
    result_anchor = '        "loss": "0.35 SmoothL1 + 0.65 centered cosine",\n'
    result_extra = '''        "loss": "0.35 SmoothL1 + 0.65 centered cosine",
        "pretraining": "mask final 15 seconds and predict hidden mid return",
        "pretrain_eval_rmse": float(pretrain_eval_rmse),
        "zero_return_rmse": float(pretrain_zero_rmse),
        "pretrain_skill": float(pretrain_skill),
        "pretrain_train_rows": int(pretrain_train_idx.size),
        "pretrain_eval_rows": int(pretrain_eval_idx.size),
'''
    source = replace_once(source, result_anchor, result_extra, "result metadata")

    ast.parse(source)
    required = [
        "PrefixPretrainDataset", "pretrain=True", "pretrain_skill",
        "PRETRAIN_MARKET_BINS = 5", "PRETRAIN_FLOW_BINS = 15",
        "static=None", "source_embedding[:, :3]",
    ]
    for token in required:
        if token not in source:
            raise RuntimeError(f"Missing required source token: {token}")

    for name in ("run_v18_predictive_pretrain_dev.py", "run.py"):
        (KERNEL / name).write_text(source, encoding="utf-8")
    notebook = {
        "cells": [{
            "id": "predictive-pretrain-dev",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (KERNEL / "run.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metadata_path = KERNEL / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
    metadata["title"] = "MultiStream Factorized Transformer MultiWindow Dev"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Prepared strict predictive-pretraining dev in the existing dual-T4 notebook")


if __name__ == "__main__":
    main()
