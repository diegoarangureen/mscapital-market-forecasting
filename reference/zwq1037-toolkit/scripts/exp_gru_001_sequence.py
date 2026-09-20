"""EXP-GRU-001: low-memory GRU over the ordered market-event cache."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-GRU-001"
SEED = 42
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 2048
HIDDEN_SIZE = 96
DROPOUT = 0.10
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
MAX_SCOUT_EPOCHS = 15
PATIENCE = 3
MIN_DELTA = 1.0e-5
RAW_CHANNEL_NAMES = [
    "transaction_avgprice",
    "transaction_volume",
    "transaction_count",
    "ask_price_1",
    "ask_volume_1",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "bid_price_2",
    "bid_volume_2",
    "seconds_before_predict",
    "has_transactions",
    "row_mask",
]
DERIVED_CHANNEL_NAMES = [
    "spread_1_bps",
    "spread_2_bps",
    "book_imbalance_1",
    "book_imbalance_2",
    "total_book_imbalance",
    "microprice_displacement_bps",
    "mid_return_bps",
    "trade_price_vs_mid_bps",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_score(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.dot(target.astype(np.float64), prediction.astype(np.float64)))
    denominator = float(np.linalg.norm(target) * np.linalg.norm(prediction))
    return 0.0 if denominator == 0.0 else numerator / denominator


def load_raw_statistics(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(11, dtype=np.float32)
    scales = np.ones(11, dtype=np.float32)
    for channel_index, channel_name in enumerate(RAW_CHANNEL_NAMES[:11]):
        statistics = json.loads(
            (cache_dir / "statistics" / f"{channel_name}.json").read_text(
                encoding="utf-8"
            )
        )
        means[channel_index] = float(statistics["mean"])
        scales[channel_index] = float(statistics["scale"])
    return means, scales


class SequenceBatchBuilder:
    """从内存映射缓存连续分批读取，并在GPU即时构建微观结构特征。"""

    def __init__(
        self,
        cache: np.memmap,
        raw_means: np.ndarray,
        raw_scales: np.ndarray,
        device: torch.device,
    ) -> None:
        self.cache = cache
        self.raw_means = torch.as_tensor(raw_means, device=device)[None, None, :]
        self.raw_scales = torch.as_tensor(raw_scales, device=device)[None, None, :]
        self.device = device

    def make(self, start: int, end: int) -> torch.Tensor:
        # 只复制一个连续小块，并取每对事件中的后一条，确保最后事件始终保留。
        # Copy one contiguous block and keep the later event of every pair.
        block = np.array(
            self.cache[start:end, :, 1::2], dtype=np.float32, copy=True
        )
        standardized = torch.from_numpy(block).to(self.device, non_blocking=False)
        standardized = standardized.transpose(1, 2).contiguous()
        standardized = torch.clamp(standardized, min=-10.0, max=10.0)
        row_mask = standardized[:, :, 13:14]

        raw_values = standardized[:, :, :11] * self.raw_scales + self.raw_means
        transaction_price = raw_values[:, :, 0]
        ask_price_1 = raw_values[:, :, 3]
        ask_volume_1 = torch.clamp(raw_values[:, :, 4], min=0.0)
        bid_price_1 = raw_values[:, :, 5]
        bid_volume_1 = torch.clamp(raw_values[:, :, 6], min=0.0)
        ask_price_2 = raw_values[:, :, 7]
        ask_volume_2 = torch.clamp(raw_values[:, :, 8], min=0.0)
        bid_price_2 = raw_values[:, :, 9]
        bid_volume_2 = torch.clamp(raw_values[:, :, 10], min=0.0)

        epsilon = 1.0e-6
        mid_price = 0.5 * (ask_price_1 + bid_price_1)
        mid_price_2 = 0.5 * (ask_price_2 + bid_price_2)
        spread_1 = (ask_price_1 - bid_price_1) / torch.clamp(mid_price.abs(), min=epsilon)
        spread_2 = (ask_price_2 - bid_price_2) / torch.clamp(mid_price_2.abs(), min=epsilon)
        imbalance_1 = (bid_volume_1 - ask_volume_1) / (
            bid_volume_1 + ask_volume_1 + epsilon
        )
        imbalance_2 = (bid_volume_2 - ask_volume_2) / (
            bid_volume_2 + ask_volume_2 + epsilon
        )
        total_imbalance = (
            bid_volume_1 + bid_volume_2 - ask_volume_1 - ask_volume_2
        ) / (
            bid_volume_1 + bid_volume_2 + ask_volume_1 + ask_volume_2 + epsilon
        )
        microprice = (
            ask_price_1 * bid_volume_1 + bid_price_1 * ask_volume_1
        ) / (bid_volume_1 + ask_volume_1 + epsilon)
        microprice_displacement = (microprice - mid_price) / torch.clamp(
            mid_price.abs(), min=epsilon
        )
        mid_return = torch.zeros_like(mid_price)
        mid_return[:, 1:] = (mid_price[:, 1:] - mid_price[:, :-1]) / torch.clamp(
            mid_price[:, :-1].abs(), min=epsilon
        )
        trade_vs_mid = (transaction_price - mid_price) / torch.clamp(
            mid_price.abs(), min=epsilon
        )
        trade_vs_mid *= standardized[:, :, 12]

        derived = torch.stack(
            [
                spread_1 * 1.0e3,
                spread_2 * 1.0e3,
                imbalance_1,
                imbalance_2,
                total_imbalance,
                microprice_displacement * 1.0e3,
                mid_return * 1.0e3,
                trade_vs_mid * 1.0e3,
            ],
            dim=2,
        )
        derived = torch.nan_to_num(derived, nan=0.0, posinf=10.0, neginf=-10.0)
        derived = torch.clamp(derived, min=-10.0, max=10.0) * row_mask
        return torch.cat([standardized, derived], dim=2)


class GRURegressor(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN_SIZE),
        )
        self.gru = nn.GRU(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE * 2),
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        row_mask = sequence[:, :, 13:14]
        projected = self.input_projection(sequence)
        recurrent_output, final_hidden = self.gru(projected)
        masked_sum = (recurrent_output * row_mask).sum(dim=1)
        valid_count = torch.clamp(row_mask.sum(dim=1), min=1.0)
        masked_mean = masked_sum / valid_count
        representation = torch.cat([final_hidden[-1], masked_mean], dim=1)
        return self.head(representation).squeeze(1)


def contiguous_blocks(start: int, end: int, batch_size: int) -> list[tuple[int, int]]:
    return [
        (block_start, min(block_start + batch_size, end))
        for block_start in range(start, end, batch_size)
    ]


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def train_epoch(
    model: GRURegressor,
    optimizer: torch.optim.Optimizer,
    batch_builder: SequenceBatchBuilder,
    target_scaled: np.ndarray,
    train_start: int,
    train_end: int,
    epoch: int,
) -> float:
    model.train()
    blocks = contiguous_blocks(train_start, train_end, BATCH_SIZE)
    np.random.default_rng(SEED + epoch).shuffle(blocks)
    total_loss = 0.0
    total_count = 0
    for start, end in blocks:
        sequence = batch_builder.make(start, end)
        target = torch.as_tensor(
            target_scaled[start:end], dtype=torch.float32, device=batch_builder.device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(sequence)
            loss = nn.functional.mse_loss(prediction.float(), target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        count = end - start
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
    return total_loss / total_count


@torch.inference_mode()
def predict_range(
    model: GRURegressor,
    batch_builder: SequenceBatchBuilder,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    model.eval()
    output = np.empty(end_index - start_index, dtype=np.float32)
    for start, end in contiguous_blocks(start_index, end_index, EVAL_BATCH_SIZE):
        sequence = batch_builder.make(start, end)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(sequence)
        output[start - start_index : end - start_index] = (
            prediction.float().cpu().numpy()
        )
    return output


def run_training_phase(
    *,
    cache: np.memmap,
    labels: pd.DataFrame,
    raw_means: np.ndarray,
    raw_scales: np.ndarray,
    train_end: int,
    validation_start: int | None,
    validation_end: int | None,
    fixed_epochs: int | None,
    checkpoint_path: Path,
    log_path: Path,
    device: torch.device,
) -> tuple[GRURegressor, SequenceBatchBuilder, int, list[dict], dict]:
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[:train_end].mean(dtype=np.float64))
    target_standard_deviation = float(target[:train_end].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_standard_deviation).astype(np.float32)
    builder = SequenceBatchBuilder(cache, raw_means, raw_scales, device)
    seed_everything(SEED)
    model = GRURegressor(len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    start_epoch = 1
    best_epoch = 1
    best_score = -np.inf
    stale_epochs = 0
    logs: list[dict] = []
    elapsed_before_resume = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["next_epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
        logs = list(checkpoint["logs"])
        elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        print(f"Resuming at epoch {start_epoch}", flush=True)

    maximum_epochs = fixed_epochs or MAX_SCOUT_EPOCHS
    started_at = time.perf_counter()
    for epoch in range(start_epoch, maximum_epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            builder,
            target_scaled,
            0,
            train_end,
            epoch,
        )
        validation_score = None
        if validation_start is not None and validation_end is not None:
            validation_prediction = predict_range(
                model, builder, validation_start, validation_end
            )
            validation_prediction *= target_standard_deviation
            validation_score = cosine_score(
                target[validation_start:validation_end], validation_prediction
            )
            if validation_score > best_score + MIN_DELTA:
                best_score = validation_score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_epoch = maximum_epochs

        elapsed_seconds = elapsed_before_resume + time.perf_counter() - started_at
        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_cosine": validation_score,
            "best_epoch": best_epoch,
            "best_score": None if not np.isfinite(best_score) else best_score,
            "elapsed_seconds": elapsed_seconds,
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False)
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "stale_epochs": stale_epochs,
                "logs": logs,
                "elapsed_seconds": elapsed_seconds,
            },
            checkpoint_path,
        )
        print(
            f"epoch={epoch:02d} loss={train_loss:.6f} "
            f"validation={validation_score} best_epoch={best_epoch}",
            flush=True,
        )
        if fixed_epochs is None and stale_epochs >= PATIENCE:
            break

    details = {
        "target_mean": target_mean,
        "target_standard_deviation": target_standard_deviation,
        "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started_at,
        "best_score": None if not np.isfinite(best_score) else best_score,
    }
    return model, builder, best_epoch, logs, details


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("EXP-GRU-001 requires a BF16-capable CUDA GPU.")

    project_dir = Path(__file__).resolve().parents[1]
    cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    run_dir = project_dir / "data" / "interim" / "sequence_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    if cache.shape != (len(labels), 14, 200):
        raise AssertionError(f"Unexpected cache shape: {cache.shape}")
    if not np.array_equal(labels["sample_id"].to_numpy(), np.arange(len(labels))):
        raise AssertionError("Label rows do not match sequence-cache positions.")
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    months = labels["month"].to_numpy()
    scout_train_end = int(np.searchsorted(months, 50, side="left"))
    scout_validation_end = int(np.searchsorted(months, 60, side="left"))
    formal_train_end = scout_validation_end
    device = torch.device("cuda")
    print(
        f"cache={cache.shape}, scout_train_end={scout_train_end}, "
        f"formal_train_end={formal_train_end}, device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    selection_path = run_dir / "epoch_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        best_epoch = int(selection["best_epoch"])
        scout_logs = pd.read_csv(run_dir / "scout_epochs.csv").to_dict(orient="records")
        scout_details = selection
        print(f"Reusing selected epoch={best_epoch}", flush=True)
    else:
        scout_model, _, best_epoch, scout_logs, scout_details = run_training_phase(
            cache=cache,
            labels=labels,
            raw_means=raw_means,
            raw_scales=raw_scales,
            train_end=scout_train_end,
            validation_start=scout_train_end,
            validation_end=scout_validation_end,
            fixed_epochs=None,
            checkpoint_path=run_dir / "scout_checkpoint.pt",
            log_path=run_dir / "scout_epochs.csv",
            device=device,
        )
        selection = {
            "best_epoch": best_epoch,
            "best_score": scout_details["best_score"],
            "elapsed_seconds": scout_details["elapsed_seconds"],
            "train_months": "0-49",
            "validation_months": "50-59",
        }
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        del scout_model
        torch.cuda.empty_cache()

    formal_model, formal_builder, _, formal_logs, formal_details = run_training_phase(
        cache=cache,
        labels=labels,
        raw_means=raw_means,
        raw_scales=raw_scales,
        train_end=formal_train_end,
        validation_start=None,
        validation_end=None,
        fixed_epochs=best_epoch,
        checkpoint_path=run_dir / "formal_checkpoint.pt",
        log_path=run_dir / "formal_epochs.csv",
        device=device,
    )
    prediction = predict_range(formal_model, formal_builder, formal_train_end, len(labels))
    prediction *= formal_details["target_standard_deviation"]
    validation_rows = labels.iloc[formal_train_end:][
        ["sample_id", "month", "target"]
    ].copy()
    tabm_baseline = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tabm-001_valid.feather"
    )
    assert_prediction_alignment(tabm_baseline, validation_rows)

    feature_columns = RAW_CHANNEL_NAMES + DERIVED_CHANNEL_NAMES
    parameters = {
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 1,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "precision": "bfloat16",
        "sequence_steps": 100,
        "downsampling": "take the later event from each consecutive pair",
    }
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "One-layer GRU over 100 ordered event steps with eight on-GPU "
            "microstructure channels; contiguous low-memory cache batches"
        ),
        model=None,
        predictions=prediction,
        validation_rows=validation_rows,
        baseline_predictions=tabm_baseline,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=formal_details["elapsed_seconds"],
        baseline_id="EXP-TABM-001",
        save_model=False,
    )
    model_path = project_dir / "outputs" / "models" / "exp-gru-001.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": formal_model.cpu().state_dict(),
            "parameters": parameters,
            "feature_columns": feature_columns,
            "raw_means": raw_means,
            "raw_scales": raw_scales,
            "selected_epoch": best_epoch,
            "target_mean": formal_details["target_mean"],
            "target_standard_deviation": formal_details[
                "target_standard_deviation"
            ],
        },
        model_path,
    )
    metadata.pop("xgboost_version", None)
    metadata.update(
        {
            "model_family": "GRU",
            "torch_version": torch.__version__,
            "training_device": "cuda-bfloat16",
            "cuda_device": torch.cuda.get_device_name(0),
            "selected_epoch": best_epoch,
            "epoch_selection_best_cosine": selection["best_score"],
            "epoch_selection_seconds": selection["elapsed_seconds"],
            "formal_training_seconds": formal_details["elapsed_seconds"],
            "model_path": str(model_path),
            "cache_path": str(cache_dir / "sequences.npy"),
            "cache_access": "numpy memory map; contiguous batches; num_workers=0",
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
