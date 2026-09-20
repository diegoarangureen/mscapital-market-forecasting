"""Fair full-200-step GRU versus masked mean/max control on two early folds."""

from __future__ import annotations

import gc
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from exp_gru_001_sequence import (
    DERIVED_CHANNEL_NAMES,
    RAW_CHANNEL_NAMES,
    load_raw_statistics,
)
from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit


EXPERIMENT_ID = "EXP-GRU-002-ORDER-CONTROL"
SEED = 42
HIDDEN_SIZE = 64
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
EPOCHS = 2
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
DROPOUT = 0.10
FIT_BATCH_SIZE = 10_000
FOLDS = {
    "train039_valid4049": (39, 40, 49),
    "train049_valid5059": (49, 50, 59),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contiguous_blocks(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(index, min(index + size, end)) for index in range(start, end, size)]


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_fold_standardization(
    cache: np.memmap, train_end: int, output_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    if output_path.exists():
        saved = np.load(output_path)
        return saved["means"], saved["scales"]
    sums = np.zeros(11, dtype=np.float64)
    squared_sums = np.zeros(11, dtype=np.float64)
    count = 0
    for start in range(0, train_end, FIT_BATCH_SIZE):
        end = min(start + FIT_BATCH_SIZE, train_end)
        values = np.asarray(cache[start:end, :11, :], dtype=np.float32)
        mask = np.asarray(cache[start:end, 13, :], dtype=np.float32) > 0.5
        sums += (values * mask[:, None, :]).sum(axis=(0, 2), dtype=np.float64)
        squared_sums += (
            np.square(values) * mask[:, None, :]
        ).sum(axis=(0, 2), dtype=np.float64)
        count += int(mask.sum())
    means = sums / count
    variances = np.maximum(squared_sums / count - np.square(means), 1.0e-8)
    scales = np.sqrt(variances)
    np.savez_compressed(
        output_path,
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        observed_event_count=np.asarray([count], dtype=np.int64),
    )
    return means.astype(np.float32), scales.astype(np.float32)


class FullSequenceBatchBuilder:
    def __init__(
        self,
        cache: np.memmap,
        raw_means: np.ndarray,
        raw_scales: np.ndarray,
        fold_means: np.ndarray,
        fold_scales: np.ndarray,
        device: torch.device,
    ) -> None:
        self.cache = cache
        self.raw_means = torch.as_tensor(raw_means, device=device)[None, None, :]
        self.raw_scales = torch.as_tensor(raw_scales, device=device)[None, None, :]
        self.fold_means = torch.as_tensor(fold_means, device=device)[None, None, :]
        self.fold_scales = torch.as_tensor(fold_scales, device=device)[None, None, :]
        self.device = device

    def make(self, start: int, end: int) -> torch.Tensor:
        block = np.array(self.cache[start:end, :, :], dtype=np.float32, copy=True)
        cached = torch.from_numpy(block).to(self.device).transpose(1, 2).contiguous()
        row_mask = cached[:, :, 13:14]
        raw_values = cached[:, :, :11] * self.raw_scales + self.raw_means

        standardized = cached.clone()
        standardized[:, :, :11] = (
            cached[:, :, :11] - self.fold_means
        ) / self.fold_scales
        standardized[:, :, :11] = torch.clamp(
            standardized[:, :, :11], min=-10.0, max=10.0
        )
        standardized *= row_mask

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
        ) / (bid_volume_1 + bid_volume_2 + ask_volume_1 + ask_volume_2 + epsilon)
        microprice = (
            ask_price_1 * bid_volume_1 + bid_price_1 * ask_volume_1
        ) / (bid_volume_1 + ask_volume_1 + epsilon)
        micro_displacement = (microprice - mid_price) / torch.clamp(
            mid_price.abs(), min=epsilon
        )
        mid_return = torch.zeros_like(mid_price)
        mid_return[:, 1:] = (mid_price[:, 1:] - mid_price[:, :-1]) / torch.clamp(
            mid_price[:, :-1].abs(), min=epsilon
        )
        trade_vs_mid = (transaction_price - mid_price) / torch.clamp(
            mid_price.abs(), min=epsilon
        )
        trade_vs_mid *= cached[:, :, 12]
        derived = torch.stack(
            [
                spread_1 * 1.0e3,
                spread_2 * 1.0e3,
                imbalance_1,
                imbalance_2,
                total_imbalance,
                micro_displacement * 1.0e3,
                mid_return * 1.0e3,
                trade_vs_mid * 1.0e3,
            ],
            dim=2,
        )
        derived = torch.nan_to_num(derived, nan=0.0, posinf=10.0, neginf=-10.0)
        derived = torch.clamp(derived, min=-10.0, max=10.0) * row_mask
        return torch.cat([standardized, derived], dim=2)


class SharedHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE * 2),
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.network(representation).squeeze(1)


class SmallGRU(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN_SIZE),
        )
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.head = SharedHead()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        row_mask = sequence[:, :, 13:14]
        projected = self.input_projection(sequence) * row_mask
        recurrent, hidden = self.gru(projected)
        masked_mean = (recurrent * row_mask).sum(dim=1) / row_mask.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([hidden[-1], masked_mean], dim=1))


class MaskedSummaryMLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN_SIZE),
        )
        self.event_network = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.SiLU(),
        )
        self.head = SharedHead()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        row_mask = sequence[:, :, 13:14]
        event = self.event_network(self.input_projection(sequence))
        masked_mean = (event * row_mask).sum(dim=1) / row_mask.sum(dim=1).clamp_min(1.0)
        masked_max = event.masked_fill(row_mask <= 0.5, -torch.inf).amax(dim=1)
        masked_max = torch.nan_to_num(masked_max, neginf=0.0)
        return self.head(torch.cat([masked_mean, masked_max], dim=1))


def train_model(
    *,
    model_name: str,
    model: nn.Module,
    builder: FullSequenceBatchBuilder,
    target_scaled: np.ndarray,
    train_end: int,
    fold_dir: Path,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    checkpoint_path = fold_dir / f"{model_name}_checkpoint.pt"
    logs: list[dict[str, float]] = []
    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=builder.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logs = checkpoint["logs"]
        start_epoch = int(checkpoint["next_epoch"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"resuming {model_name} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        blocks = contiguous_blocks(0, train_end, BATCH_SIZE)
        np.random.default_rng(SEED + epoch).shuffle(blocks)
        total_loss = 0.0
        total_count = 0
        for start, end in blocks:
            sequence = builder.make(start, end)
            target = torch.as_tensor(
                target_scaled[start:end], dtype=torch.float32, device=builder.device
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
        logs.append({"epoch": epoch, "train_mse": total_loss / total_count})
        pd.DataFrame(logs).to_csv(fold_dir / f"{model_name}_training.csv", index=False)
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "logs": logs,
                "next_epoch": epoch + 1,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_path,
        )
        print(
            f"{fold_dir.name} {model_name} epoch={epoch}/{EPOCHS} "
            f"loss={logs[-1]['train_mse']:.7f}",
            flush=True,
        )
    return logs


@torch.inference_mode()
def predict_range(
    model: nn.Module,
    builder: FullSequenceBatchBuilder,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    model.eval()
    output = np.empty(end_index - start_index, dtype=np.float32)
    for start, end in contiguous_blocks(start_index, end_index, EVAL_BATCH_SIZE):
        sequence = builder.make(start, end)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(sequence)
        output[start - start_index : end - start_index] = prediction.float().cpu().numpy()
    return output


def run_fold(
    *,
    project_dir: Path,
    run_dir: Path,
    fold_name: str,
    train_end_month: int,
    valid_start_month: int,
    valid_end_month: int,
    cache: np.memmap,
    labels: pd.DataFrame,
    raw_means: np.ndarray,
    raw_scales: np.ndarray,
    device: torch.device,
) -> dict:
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    months = labels["month"].to_numpy()
    train_end = int(np.searchsorted(months, train_end_month + 1, side="left"))
    valid_start = int(np.searchsorted(months, valid_start_month, side="left"))
    valid_end = int(np.searchsorted(months, valid_end_month + 1, side="left"))
    if train_end != valid_start:
        raise AssertionError("Expected contiguous forward split.")
    fold_means, fold_scales = fit_fold_standardization(
        cache, train_end, fold_dir / "fold_standardization.npz"
    )
    builder = FullSequenceBatchBuilder(
        cache, raw_means, raw_scales, fold_means, fold_scales, device
    )
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[:train_end].mean(dtype=np.float64))
    target_std = float(target[:train_end].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_std).astype(np.float32)

    predictions = {}
    model_details = {}
    for model_name, model_class in (("gru", SmallGRU), ("summary", MaskedSummaryMLP)):
        seed_everything(SEED)
        model = model_class(len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)).to(device)
        logs = train_model(
            model_name=model_name,
            model=model,
            builder=builder,
            target_scaled=target_scaled,
            train_end=train_end,
            fold_dir=fold_dir,
        )
        prediction = predict_range(model, builder, valid_start, valid_end).astype(np.float64)
        predictions[model_name] = prediction * target_std
        model_details[model_name] = {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "logs": logs,
        }
        del model
        torch.cuda.empty_cache()
        gc.collect()

    static_frame = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-016-RELATIVE-SCALE-ADD12"
        / fold_name
        / "validation_predictions.feather"
    )
    validation_rows = labels.iloc[valid_start:valid_end]
    if not np.array_equal(
        static_frame["sample_id"].to_numpy(), validation_rows["sample_id"].to_numpy()
    ):
        raise AssertionError("Static and sequence validation IDs differ.")
    static = static_frame["b001_trim1"].to_numpy(dtype=np.float64)
    predictions["static"] = static
    predictions["static90_gru10"] = 0.90 * unit(static) + 0.10 * unit(predictions["gru"])
    predictions["static90_summary10"] = 0.90 * unit(static) + 0.10 * unit(
        predictions["summary"]
    )
    validation_target = validation_rows["target"].to_numpy(dtype=np.float64)
    validation_months = validation_rows["month"].to_numpy()
    metrics = {
        name: evaluate(
            validation_target,
            prediction,
            validation_months,
            valid_start_month,
            valid_end_month,
        )
        for name, prediction in predictions.items()
    }
    for name in ("static90_gru10", "static90_summary10"):
        for key in ("overall", "monthly_std", "monthly_worst", "monthly_q25"):
            metrics[name][f"{key}_delta"] = metrics[name][key] - metrics["static"][key]

    output = validation_rows[["sample_id", "month", "target"]].copy()
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_feather(fold_dir / "validation_predictions.feather")
    summary = []
    for name, values in metrics.items():
        summary.append(
            {
                "candidate": name,
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"monthly", "first_half_months", "second_half_months"}
                },
            }
        )
    pd.DataFrame(summary).to_csv(fold_dir / "summary.csv", index=False)
    pd.DataFrame(
        {
            "month": list(range(valid_start_month, valid_end_month + 1)),
            **{
                name: [row["cosine"] for row in values["monthly"]]
                for name, values in metrics.items()
            },
        }
    ).to_csv(fold_dir / "monthly_cosine.csv", index=False)
    result = {
        "fold": fold_name,
        "train_months": f"0-{train_end_month}",
        "validation_months": f"{valid_start_month}-{valid_end_month}",
        "train_rows": train_end,
        "validation_rows": valid_end - valid_start,
        "target_mean": target_mean,
        "target_std": target_std,
        "model_details": model_details,
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "sequence_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    if cache.shape != (len(labels), 14, 200):
        raise AssertionError("Sequence cache and labels differ.")
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    results = {}
    for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
        results[fold_name] = run_fold(
            project_dir=project_dir,
            run_dir=run_dir,
            fold_name=fold_name,
            train_end_month=train_end,
            valid_start_month=valid_start,
            valid_end_month=valid_end,
            cache=cache,
            labels=labels,
            raw_means=raw_means,
            raw_scales=raw_scales,
            device=torch.device("cuda"),
        )
    deltas = [
        results[fold]["metrics"]["static90_gru10"]["overall_delta"] for fold in FOLDS
    ]
    summary_deltas = [
        results[fold]["metrics"]["static90_summary10"]["overall_delta"] for fold in FOLDS
    ]
    decision = {
        "gru_both_windows_positive": bool(all(delta > 0 for delta in deltas)),
        "gru_average_delta": float(np.mean(deltas)),
        "summary_average_delta": float(np.mean(summary_deltas)),
        "passes_stage1_threshold": bool(
            all(delta > 0 for delta in deltas)
            and np.mean(deltas) >= 0.0015
            and np.mean(deltas) > np.mean(summary_deltas)
        ),
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "test incremental value of event order against an unordered control",
        "fixed_parameters": {
            "seed": SEED,
            "steps": 200,
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "loss": "training-fold globally standardized target MSE",
            "blend": "90% relative319 static + 10% sequence model after L2 normalization",
            "training_preprocessing": "raw-channel standardization fitted separately in each training fold",
        },
        "results": results,
        "decision": decision,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
