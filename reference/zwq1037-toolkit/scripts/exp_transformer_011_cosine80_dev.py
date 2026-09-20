"""Test SmoothL1 plus centered cosine loss on the existing Joint-Transformer."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from torch import nn

import exp_gru_003_strong_joint as gru
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import QuantileBatchPreprocessor, evaluate, fit_quantile_knots
from exp_tabm_014_quantile_stage_curves import unit
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS


EXPERIMENT_ID = "EXP-TRANSFORMER-011-COSINE80-DEV"
SEED = 42
TRAIN_END_MONTH = 49
VALID_START_MONTH = 50
VALID_END_MONTH = 59
FOLD_NAME = "train049_valid5059"
EPOCHS = 6
EVALUATION_EPOCHS = {1, 2, 3, 4, 5, 6}
EFFECTIVE_BATCH_SIZE = 512
MICRO_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
D_MODEL = 96
NUM_HEADS = 4
NUM_LAYERS = 2
FEEDFORWARD_SIZE = 192
DROPOUT = 0.10
RECENT_EVENT_COUNT = 32
MAX_LEARNING_RATE = 1.0e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
SMOOTH_L1_WEIGHT = 0.20
COSINE_WEIGHT = 0.80


def centered_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """优化一个 batch 内预测与目标的共同方向。 / Optimize batch-wise direction."""
    prediction_centered = prediction.float() - prediction.float().mean()
    target_centered = target.float() - target.float().mean()
    numerator = torch.sum(prediction_centered * target_centered)
    denominator = torch.linalg.vector_norm(prediction_centered) * torch.linalg.vector_norm(target_centered)
    return 1.0 - numerator / denominator.clamp_min(1.0e-8)


class TransformerSequenceEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, D_MODEL),
            nn.SiLU(),
            nn.LayerNorm(D_MODEL),
        )
        self.position_embedding = nn.Parameter(torch.empty(1, 200, D_MODEL))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_SIZE,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=NUM_LAYERS, norm=nn.LayerNorm(D_MODEL),
            enable_nested_tensor=False,
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.Tanh(),
            nn.Linear(D_MODEL // 2, 1),
        )

    @property
    def output_size(self) -> int:
        return D_MODEL * 4

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence, valid_prefix, lengths = gru.left_align_valid_events(sequence)
        time_steps = sequence.shape[1]
        tokens = self.input_projection(sequence)
        tokens = tokens + self.position_embedding[:, :time_steps]
        encoded = self.transformer(tokens, src_key_padding_mask=~valid_prefix)
        mask = valid_prefix[:, :, None]
        encoded = encoded * mask

        positions = torch.arange(time_steps, device=sequence.device)[None, :]
        last = encoded[torch.arange(len(encoded), device=sequence.device), lengths - 1]
        full_mean = encoded.sum(dim=1) / lengths[:, None]
        recent_start = torch.clamp(lengths - RECENT_EVENT_COUNT, min=0)
        recent_mask = (positions >= recent_start[:, None]) & valid_prefix
        recent_count = recent_mask.sum(dim=1).clamp_min(1)
        recent_mean = (encoded * recent_mask[:, :, None]).sum(dim=1) / recent_count[:, None]
        attention_logits = self.attention_pool(encoded).squeeze(2)
        attention_logits = attention_logits.masked_fill(~valid_prefix, -torch.inf)
        attention_weights = torch.softmax(attention_logits.float(), dim=1).to(encoded.dtype)
        attention_mean = (encoded * attention_weights[:, :, None]).sum(dim=1)
        return torch.cat([last, full_mean, recent_mean, attention_mean], dim=1)


class JointTransformerRegressor(nn.Module):
    def __init__(self, sequence_input_size: int, static_input_size: int) -> None:
        super().__init__()
        self.encoder = TransformerSequenceEncoder(sequence_input_size)
        self.sequence_adapter = nn.Sequential(
            nn.Linear(self.encoder.output_size, D_MODEL, bias=False),
            nn.SiLU(),
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(static_input_size, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, D_MODEL),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL * 2),
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        sequence_representation = self.sequence_adapter(self.encoder(sequence))
        static_representation = self.static_adapter(static)
        return self.head(
            torch.cat([static_representation, sequence_representation], dim=1)
        ).squeeze(1)


def check_padding_invariance(
    model: JointTransformerRegressor,
    input_size: int,
    static_input_size: int,
    device: torch.device,
) -> None:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(20260912)
    events = torch.randn((1, 7, input_size), generator=generator, device=device)
    events[:, :, 13] = 1.0
    padded = torch.zeros((1, 13, input_size), device=device)
    padded[:, -7:] = events
    static = torch.randn((1, static_input_size), generator=generator, device=device)
    with torch.inference_mode():
        left = model(events, static)
        right = model(padded, static)
    if not torch.allclose(left.float(), right.float(), atol=3.0e-5, rtol=3.0e-5):
        raise AssertionError(f"Padding changes prediction: {left.item()} vs {right.item()}")

def learning_rate(epoch: int) -> float:
    progress = (epoch - 1) / max(EPOCHS - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + (MAX_LEARNING_RATE - MIN_LEARNING_RATE) * cosine


@torch.inference_mode()
def predict(
    model: JointTransformerRegressor,
    builder: FullSequenceBatchBuilder,
    static_features: np.ndarray,
    preprocessor: QuantileBatchPreprocessor,
    start: int,
    end: int,
    target_scale: float,
) -> np.ndarray:
    model.eval()
    output = np.empty(end - start, dtype=np.float32)
    for batch_start, batch_end in gru.blocks(start, end, EVAL_BATCH_SIZE):
        sequence = builder.make(batch_start, batch_end)
        static = preprocessor.transform(static_features[batch_start:batch_end])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(sequence, static)
        output[batch_start - start:batch_end - start] = (
            prediction.float().cpu().numpy() * target_scale
        )
    return output


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
    months = labels["month"].to_numpy()
    train_end = int(np.searchsorted(months, TRAIN_END_MONTH + 1, side="left"))
    valid_start = int(np.searchsorted(months, VALID_START_MONTH, side="left"))
    valid_end = int(np.searchsorted(months, VALID_END_MONTH + 1, side="left"))
    if train_end != valid_start or cache.shape != (len(labels), 14, 200):
        raise AssertionError("Sequence cache or forward split differs.")
    gru.check_cache_layout(cache)

    baseline = gru.load_baseline(
        project_dir, labels.iloc[valid_start:valid_end]["sample_id"].to_numpy()
    )
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    fold_means, fold_scales = fit_fold_standardization(
        cache, train_end, run_dir / "fold_standardization.npz"
    )
    builder = FullSequenceBatchBuilder(
        cache, raw_means, raw_scales, fold_means, fold_scales, device
    )
    static_features, feature_columns = gru.load_relative319(project_dir, labels)
    if len(feature_columns) != 319 or feature_columns[-len(RELATIVE_COLUMNS):] != RELATIVE_COLUMNS:
        raise AssertionError("Relative319 schema differs.")
    preprocessing_path = run_dir / "quantile_preprocessing.npz"
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
    else:
        train_indices = np.arange(train_end, dtype=np.int64)
        knots, medians, missing_columns = fit_quantile_knots(
            static_features, train_indices, SEED
        )
        np.savez_compressed(
            preprocessing_path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray(feature_columns),
        )
    preprocessor = QuantileBatchPreprocessor(knots, medians, missing_columns, device)
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[:train_end].mean(dtype=np.float64))
    target_scale = float(target[:train_end].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_scale).astype(np.float32)
    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)
    gru.seed_everything(SEED)
    model = JointTransformerRegressor(input_size, preprocessor.output_dimension).to(device)
    check_padding_invariance(model, input_size, preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    logs: list[dict] = []
    metrics_by_epoch = {}
    start_epoch = 1
    elapsed_before = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logs = list(checkpoint["logs"])
        metrics_by_epoch = dict(checkpoint["metrics_by_epoch"])
        start_epoch = int(checkpoint["next_epoch"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        print(f"resuming at epoch {start_epoch}", flush=True)

    validation_rows = labels.iloc[valid_start:valid_end]
    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        rate = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        train_blocks = gru.blocks(0, train_end, EFFECTIVE_BATCH_SIZE)
        np.random.default_rng(SEED + epoch).shuffle(train_blocks)
        loss_sum = 0.0
        row_count = 0
        for block_start, block_end in train_blocks:
            optimizer.zero_grad(set_to_none=True)
            block_count = block_end - block_start
            for micro_start in range(block_start, block_end, MICRO_BATCH_SIZE):
                micro_end = min(micro_start + MICRO_BATCH_SIZE, block_end)
                sequence = builder.make(micro_start, micro_end)
                static = preprocessor.transform(static_features[micro_start:micro_end])
                batch_target = torch.as_tensor(target_scaled[micro_start:micro_end], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(sequence, static)
                    smooth_l1 = nn.functional.smooth_l1_loss(prediction.float(), batch_target)
                    cosine = centered_cosine_loss(prediction, batch_target)
                    loss = SMOOTH_L1_WEIGHT * smooth_l1 + COSINE_WEIGHT * cosine
                count = micro_end - micro_start
                (loss * (count / block_count)).backward()
                loss_sum += float(loss.detach().cpu()) * count
                row_count += count
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        elapsed = elapsed_before + time.perf_counter() - started
        log = {
            "epoch": epoch,
            "learning_rate": rate,
            "train_loss": loss_sum / row_count,
            "elapsed_seconds": elapsed,
        }
        logs.append(log)
        if epoch in EVALUATION_EPOCHS:
            candidate = predict(
                model, builder, static_features, preprocessor,
                valid_start, valid_end, target_scale,
            ).astype(np.float64)
            blended = 0.90 * unit(baseline) + 0.10 * unit(candidate)
            validation_target = validation_rows["target"].to_numpy(dtype=np.float64)
            validation_months = validation_rows["month"].to_numpy()
            standalone_metrics = evaluate(
                validation_target, candidate, validation_months,
                VALID_START_MONTH, VALID_END_MONTH,
            )
            blend_metrics = evaluate(
                validation_target, blended, validation_months,
                VALID_START_MONTH, VALID_END_MONTH,
            )
            baseline_metrics = evaluate(
                validation_target, baseline, validation_months,
                VALID_START_MONTH, VALID_END_MONTH,
            )
            blend_metrics["overall_delta_vs_b001"] = (
                blend_metrics["overall"] - baseline_metrics["overall"]
            )
            metrics_by_epoch[str(epoch)] = {
                "standalone": standalone_metrics,
                "blend90": blend_metrics,
                "baseline": baseline_metrics,
            }
            output = validation_rows[["sample_id", "month", "target"]].copy()
            output["b001_trim1"] = baseline
            output["prediction"] = candidate
            output["blend90"] = blended
            output.to_feather(run_dir / f"validation_predictions_epoch{epoch:02d}.feather")
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch},
                run_dir / f"model_epoch{epoch:02d}.pt",
            )
            print(
                f"epoch={epoch}/{EPOCHS} loss={log['train_loss']:.7f} "
                f"cos={standalone_metrics['overall']:.7f} "
                f"blend_delta={blend_metrics['overall_delta_vs_b001']:+.7f}",
                flush=True,
            )
        else:
            print(f"epoch={epoch}/{EPOCHS} loss={log['train_loss']:.7f}", flush=True)
        pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
        gru.atomic_save(
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
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "seed": SEED,
        "train_months": f"0-{TRAIN_END_MONTH}",
        "validation_months": f"{VALID_START_MONTH}-{VALID_END_MONTH}",
        "training_rows": train_end,
        "validation_rows": valid_end - valid_start,
        "feature_count": len(feature_columns),
        "sequence_input_channels": input_size,
        "architecture": {
            "d_model": D_MODEL,
            "heads": NUM_HEADS,
            "layers": NUM_LAYERS,
            "feedforward_size": FEEDFORWARD_SIZE,
            "position_encoding": "learned relative event positions after left alignment",
            "pooling": "last + full mean + recent32 mean + attention",
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "loss": {"smooth_l1_weight": SMOOTH_L1_WEIGHT, "centered_cosine_weight": COSINE_WEIGHT},
        "logs": logs,
        "metrics_by_epoch": metrics_by_epoch,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, static_features, preprocessor
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()



