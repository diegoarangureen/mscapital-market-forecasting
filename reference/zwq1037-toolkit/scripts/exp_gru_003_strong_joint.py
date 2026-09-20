"""Train a stronger padding-aware GRU and a joint GRU + relative319 model."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import exp_tabm_001_exp053r_features as tabm
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import QuantileBatchPreprocessor, evaluate, fit_quantile_knots
from exp_tabm_014_quantile_stage_curves import unit
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features


EXPERIMENT_ID = "EXP-GRU-003-STRONG-JOINT-DEV"
SEED = 42
TRAIN_END_MONTH = 49
VALID_START_MONTH = 50
VALID_END_MONTH = 59
FOLD_NAME = "train049_valid5059"
EPOCHS = 8
EVALUATION_EPOCHS = {2, 4, 6, 8}
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
HIDDEN_SIZE = 96
RECENT_EVENT_COUNT = 32
DROPOUT = 0.10
MAX_LEARNING_RATE = 1.0e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def blocks(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(index, min(index + size, end)) for index in range(start, end, size)]


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def learning_rate(epoch: int) -> float:
    progress = (epoch - 1) / max(EPOCHS - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + (MAX_LEARNING_RATE - MIN_LEARNING_RATE) * cosine


def left_align_valid_events(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move each contiguous valid suffix to the front while preserving event order."""
    mask = sequence[:, :, 13] > 0.5
    lengths = mask.sum(dim=1, dtype=torch.int64)
    if torch.any(lengths <= 0):
        raise ValueError("Every sequence must contain at least one valid event.")
    time_steps = sequence.shape[1]
    positions = torch.arange(time_steps, device=sequence.device)[None, :]
    start = time_steps - lengths
    source = (start[:, None] + positions).clamp(max=time_steps - 1)
    source = source[:, :, None].expand(-1, -1, sequence.shape[2])
    aligned = torch.gather(sequence, dim=1, index=source)
    valid_prefix = positions < lengths[:, None]
    aligned = aligned * valid_prefix[:, :, None]
    aligned[:, :, 13] = valid_prefix
    return aligned, valid_prefix, lengths


class StrongSequenceEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN_SIZE),
        )
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE // 2, 1),
        )

    @property
    def output_size(self) -> int:
        return HIDDEN_SIZE * 4

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence, valid_prefix, lengths = left_align_valid_events(sequence)
        projected = self.input_projection(sequence) * valid_prefix[:, :, None]
        packed = pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_recurrent, hidden = self.gru(packed)
        recurrent, _ = pad_packed_sequence(
            packed_recurrent,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        mask = valid_prefix[:, :, None]
        full_mean = (recurrent * mask).sum(dim=1) / lengths[:, None]

        positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        recent_start = torch.clamp(lengths - RECENT_EVENT_COUNT, min=0)
        recent_mask = (positions >= recent_start[:, None]) & valid_prefix
        recent_count = recent_mask.sum(dim=1).clamp_min(1)
        recent_mean = (recurrent * recent_mask[:, :, None]).sum(dim=1) / recent_count[:, None]

        attention_logits = self.attention(recurrent).squeeze(2)
        attention_logits = attention_logits.masked_fill(~valid_prefix, -torch.inf)
        attention_weights = torch.softmax(attention_logits.float(), dim=1).to(recurrent.dtype)
        attention_pool = (recurrent * attention_weights[:, :, None]).sum(dim=1)
        return torch.cat([hidden[-1], full_mean, recent_mean, attention_pool], dim=1)


class StrongGRURegressor(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = StrongSequenceEncoder(input_size)
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_size),
            nn.Linear(self.encoder.output_size, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(sequence)).squeeze(1)


class JointRegressor(nn.Module):
    """Use the same module graph for the static control and joint candidate."""
    def __init__(self, sequence_input_size: int, static_input_size: int, use_sequence: bool) -> None:
        super().__init__()
        self.use_sequence = use_sequence
        self.encoder = StrongSequenceEncoder(sequence_input_size)
        self.sequence_adapter = nn.Sequential(
            nn.Linear(self.encoder.output_size, HIDDEN_SIZE, bias=False),
            nn.SiLU(),
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(static_input_size, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, HIDDEN_SIZE),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE * 2),
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, sequence: torch.Tensor | None, static: torch.Tensor) -> torch.Tensor:
        static_representation = self.static_adapter(static)
        if self.use_sequence:
            if sequence is None:
                raise ValueError("Joint model requires sequence input.")
            sequence_representation = self.sequence_adapter(self.encoder(sequence))
        else:
            sequence_representation = torch.zeros_like(static_representation)
        combined = torch.cat([static_representation, sequence_representation], dim=1)
        return self.head(combined).squeeze(1)


def check_cache_layout(cache: np.memmap) -> None:
    indices = np.linspace(0, len(cache) - 1, num=4096, dtype=np.int64)
    mask = np.asarray(cache[indices, 13, :], dtype=np.float32) > 0.5
    if np.any(mask[:, :-1] & ~mask[:, 1:]):
        raise AssertionError("Expected left padding followed by one contiguous valid suffix.")
    if np.any(mask.sum(axis=1) == 0):
        raise AssertionError("Empty sequences found.")


def check_padding_invariance(model: nn.Module, input_size: int, device: torch.device) -> None:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(20260912)
    events = torch.randn((1, 7, input_size), generator=generator, device=device)
    events[:, :, 13] = 1.0
    padded = torch.zeros((1, 13, input_size), device=device)
    padded[:, -7:] = events
    with torch.inference_mode():
        if isinstance(model, JointRegressor):
            static = torch.randn((1, model.static_adapter[0].in_features), generator=generator, device=device)
            left = model(events, static)
            right = model(padded, static)
        else:
            left = model(events)
            right = model(padded)
    if not torch.allclose(left.float(), right.float(), atol=2.0e-5, rtol=2.0e-5):
        raise AssertionError(f"Padding changes prediction: {left.item()} vs {right.item()}")


def load_baseline(project_dir: Path, validation_ids: np.ndarray) -> np.ndarray:
    path = (
        project_dir / "data" / "interim" / "tree_experiments"
        / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / FOLD_NAME
        / "validation_predictions.feather"
    )
    frame = pd.read_feather(path, columns=["sample_id", "b001_trim1"])
    if not np.array_equal(frame["sample_id"].to_numpy(), validation_ids):
        raise AssertionError("Relative319 baseline IDs differ from validation rows.")
    return frame["b001_trim1"].to_numpy(dtype=np.float64)


@torch.inference_mode()
def predict_model(
    model: nn.Module,
    builder: FullSequenceBatchBuilder,
    start: int,
    end: int,
    static_features: np.ndarray | None,
    static_preprocessor: QuantileBatchPreprocessor | None,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.empty(end - start, dtype=np.float32)
    for batch_start, batch_end in blocks(start, end, EVAL_BATCH_SIZE):
        sequence = None
        if not isinstance(model, JointRegressor) or model.use_sequence:
            sequence = builder.make(batch_start, batch_end)
        static = None
        if static_features is not None:
            static = static_preprocessor.transform(static_features[batch_start:batch_end])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(sequence, static)
        output[batch_start - start:batch_end - start] = prediction.float().cpu().numpy()
    return output


def train_candidate(
    *,
    name: str,
    model: nn.Module,
    builder: FullSequenceBatchBuilder,
    target_scaled: np.ndarray,
    target_scale: float,
    train_end: int,
    valid_start: int,
    valid_end: int,
    validation_rows: pd.DataFrame,
    baseline: np.ndarray,
    run_dir: Path,
    static_features: np.ndarray | None,
    static_preprocessor: QuantileBatchPreprocessor | None,
    device: torch.device,
) -> dict:
    candidate_dir = run_dir / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = candidate_dir / "checkpoint.pt"
    result_path = candidate_dir / "result.json"
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logs: list[dict] = []
    metrics_by_epoch: dict[str, dict] = {}
    start_epoch = 1
    elapsed_before_resume = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logs = checkpoint["logs"]
        metrics_by_epoch = checkpoint["metrics_by_epoch"]
        start_epoch = int(checkpoint["next_epoch"])
        elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"resuming {name} at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        rate = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        train_blocks = blocks(0, train_end, BATCH_SIZE)
        np.random.default_rng(SEED + epoch).shuffle(train_blocks)
        total_loss = 0.0
        total_count = 0
        for batch_start, batch_end in train_blocks:
            sequence = None
            if not isinstance(model, JointRegressor) or model.use_sequence:
                sequence = builder.make(batch_start, batch_end)
            static = None
            if static_features is not None:
                static = static_preprocessor.transform(static_features[batch_start:batch_end])
            target = torch.as_tensor(target_scaled[batch_start:batch_end], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(sequence, static)
                loss = nn.functional.mse_loss(prediction.float(), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = batch_end - batch_start
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        elapsed = elapsed_before_resume + time.perf_counter() - started
        log = {"epoch": epoch, "learning_rate": rate, "train_mse": total_loss / total_count, "elapsed_seconds": elapsed}
        logs.append(log)

        if epoch in EVALUATION_EPOCHS:
            prediction = predict_model(
                model, builder, valid_start, valid_end,
                static_features, static_preprocessor, device,
            ).astype(np.float64) * target_scale
            blend = 0.90 * unit(baseline) + 0.10 * unit(prediction)
            target = validation_rows["target"].to_numpy(dtype=np.float64)
            months = validation_rows["month"].to_numpy()
            standalone_metrics = evaluate(target, prediction, months, VALID_START_MONTH, VALID_END_MONTH)
            blend_metrics = evaluate(target, blend, months, VALID_START_MONTH, VALID_END_MONTH)
            baseline_metrics = evaluate(target, baseline, months, VALID_START_MONTH, VALID_END_MONTH)
            blend_metrics["overall_delta_vs_b001"] = blend_metrics["overall"] - baseline_metrics["overall"]
            metrics_by_epoch[str(epoch)] = {
                "standalone": standalone_metrics,
                "blend90": blend_metrics,
                "baseline": baseline_metrics,
            }
            output = validation_rows[["sample_id", "month", "target"]].copy()
            output["b001_trim1"] = baseline
            output["prediction"] = prediction
            output["blend90"] = blend
            output.to_feather(candidate_dir / f"validation_predictions_epoch{epoch:02d}.feather")
            atomic_save({"model_state": model.state_dict(), "epoch": epoch}, candidate_dir / f"model_epoch{epoch:02d}.pt")
            print(
                f"{name} epoch={epoch}/{EPOCHS} loss={log['train_mse']:.7f} "
                f"cos={standalone_metrics['overall']:.7f} blend={blend_metrics['overall']:.7f} "
                f"delta={blend_metrics['overall_delta_vs_b001']:+.7f}",
                flush=True,
            )
        else:
            print(f"{name} epoch={epoch}/{EPOCHS} loss={log['train_mse']:.7f}", flush=True)

        pd.DataFrame(logs).to_csv(candidate_dir / "training.csv", index=False)
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "metrics_by_epoch": metrics_by_epoch,
                "elapsed_seconds": elapsed,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_path,
        )

    result = {
        "name": name,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "logs": logs,
        "metrics_by_epoch": metrics_by_epoch,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_relative319(project_dir: Path, labels: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    model_data, base_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, model_data)
    columns = [*base_columns, *RELATIVE_COLUMNS]
    if len(columns) != 319 or len(set(columns)) != 319:
        raise AssertionError("Expected 319 unique relative features.")
    if not np.array_equal(model_data["sample_id"].to_numpy(), labels["sample_id"].to_numpy()):
        raise AssertionError("Static features and sequence labels are not aligned.")
    features = model_data[columns].to_numpy(dtype=np.float32, copy=True)
    del model_data
    gc.collect()
    return features, columns


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    device = torch.device("cuda")
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
    months = labels["month"].to_numpy()
    if np.any(months[1:] < months[:-1]):
        raise AssertionError("Expected sample-ID order to preserve month order.")
    train_end = int(np.searchsorted(months, TRAIN_END_MONTH + 1, side="left"))
    valid_start = int(np.searchsorted(months, VALID_START_MONTH, side="left"))
    valid_end = int(np.searchsorted(months, VALID_END_MONTH + 1, side="left"))
    if train_end != valid_start:
        raise AssertionError("Expected contiguous forward split.")
    validation_rows = labels.iloc[valid_start:valid_end].copy()
    baseline = load_baseline(project_dir, validation_rows["sample_id"].to_numpy())

    check_cache_layout(cache)
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    fold_means, fold_scales = fit_fold_standardization(cache, train_end, run_dir / "fold_standardization.npz")
    builder = FullSequenceBatchBuilder(cache, raw_means, raw_scales, fold_means, fold_scales, device)
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[:train_end].mean(dtype=np.float64))
    target_scale = float(target[:train_end].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_scale).astype(np.float32)
    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)

    results = {}
    seed_everything(SEED)
    strong = StrongGRURegressor(input_size).to(device)
    check_padding_invariance(strong, input_size, device)
    results["strong_gru"] = train_candidate(
        name="strong_gru", model=strong, builder=builder,
        target_scaled=target_scaled, target_scale=target_scale,
        train_end=train_end, valid_start=valid_start, valid_end=valid_end,
        validation_rows=validation_rows, baseline=baseline, run_dir=run_dir,
        static_features=None, static_preprocessor=None, device=device,
    )
    del strong
    torch.cuda.empty_cache()
    gc.collect()

    static_features, feature_columns = load_relative319(project_dir, labels)
    train_indices = np.arange(train_end, dtype=np.int64)
    preprocessing_path = run_dir / "quantile_preprocessing.npz"
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots, medians, missing_columns = saved["knots"], saved["medians"], saved["missing_columns"]
    else:
        knots, medians, missing_columns = fit_quantile_knots(static_features, train_indices, SEED)
        np.savez_compressed(preprocessing_path, knots=knots, medians=medians,
                            missing_columns=missing_columns, feature_columns=np.asarray(feature_columns))
    preprocessor = QuantileBatchPreprocessor(knots, medians, missing_columns, device)

    for name, use_sequence in (("static_control", False), ("joint_gru319", True)):
        seed_everything(SEED)
        model = JointRegressor(input_size, preprocessor.output_dimension, use_sequence).to(device)
        check_padding_invariance(model, input_size, device)
        results[name] = train_candidate(
            name=name, model=model, builder=builder,
            target_scaled=target_scaled, target_scale=target_scale,
            train_end=train_end, valid_start=valid_start, valid_end=valid_end,
            validation_rows=validation_rows, baseline=baseline, run_dir=run_dir,
            static_features=static_features, static_preprocessor=preprocessor, device=device,
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()

    comparisons = []
    for epoch in sorted(EVALUATION_EPOCHS):
        static_metrics = results["static_control"]["metrics_by_epoch"][str(epoch)]
        joint_metrics = results["joint_gru319"]["metrics_by_epoch"][str(epoch)]
        comparisons.append({
            "epoch": epoch,
            "strong_gru_cosine": results["strong_gru"]["metrics_by_epoch"][str(epoch)]["standalone"]["overall"],
            "strong_gru_blend_delta": results["strong_gru"]["metrics_by_epoch"][str(epoch)]["blend90"]["overall_delta_vs_b001"],
            "static_control_cosine": static_metrics["standalone"]["overall"],
            "static_control_blend_delta": static_metrics["blend90"]["overall_delta_vs_b001"],
            "joint_cosine": joint_metrics["standalone"]["overall"],
            "joint_blend_delta": joint_metrics["blend90"]["overall_delta_vs_b001"],
            "joint_minus_static_cosine": joint_metrics["standalone"]["overall"] - static_metrics["standalone"]["overall"],
            "joint_minus_static_blend": joint_metrics["blend90"]["overall"] - static_metrics["blend90"]["overall"],
        })
    pd.DataFrame(comparisons).to_csv(run_dir / "comparison.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "seed": SEED,
        "train_months": f"0-{TRAIN_END_MONTH}",
        "validation_months": f"{VALID_START_MONTH}-{VALID_END_MONTH}",
        "training_rows": train_end,
        "validation_rows": valid_end - valid_start,
        "fixed_epochs": EPOCHS,
        "evaluation_epochs": sorted(EVALUATION_EPOCHS),
        "sequence_input_channels": input_size,
        "hidden_size": HIDDEN_SIZE,
        "recent_event_count": RECENT_EVENT_COUNT,
        "feature_count": len(feature_columns),
        "static_input_after_missing_indicators": preprocessor.output_dimension,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "models": results,
        "comparisons": comparisons,
        "limitations": [
            "The existing float16 cache was encoded with unlabeled 0-59 statistics; fold re-standardization is applied but quantization cannot be exactly reversed.",
            "The 50-59 window is development data for choosing the next fixed configuration; it is not final confirmation.",
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
