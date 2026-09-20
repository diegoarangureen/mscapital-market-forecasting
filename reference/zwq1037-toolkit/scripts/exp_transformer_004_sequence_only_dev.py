"""Train a Transformer using only per-event sequence channels, without static features."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from torch import nn

import exp_transformer_001_joint_dev as base
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit


EXPERIMENT_ID = "EXP-TRANSFORMER-004-SEQUENCE-ONLY-DEV"
SEED = 42
TRAIN_END_MONTH = 49
VALID_START_MONTH = 50
VALID_END_MONTH = 59
FOLD_NAME = "train049_valid5059"
EPOCHS = 6
EVALUATION_EPOCHS = {2, 4, 6}


class SequenceOnlyTransformer(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = base.TransformerSequenceEncoder(input_size)
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_size),
            nn.Linear(self.encoder.output_size, base.D_MODEL),
            nn.SiLU(),
            nn.Dropout(base.DROPOUT),
            nn.Linear(base.D_MODEL, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(sequence)).squeeze(1)


def check_padding_invariance(
    model: SequenceOnlyTransformer, input_size: int, device: torch.device
) -> None:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(20260912)
    events = torch.randn((1, 7, input_size), generator=generator, device=device)
    events[:, :, 13] = 1.0
    padded = torch.zeros((1, 13, input_size), device=device)
    padded[:, -7:] = events
    with torch.inference_mode():
        left = model(events)
        right = model(padded)
    if not torch.allclose(left.float(), right.float(), atol=3.0e-5, rtol=3.0e-5):
        raise AssertionError(f"Padding changes prediction: {left.item()} vs {right.item()}")


@torch.inference_mode()
def predict(
    model: SequenceOnlyTransformer,
    builder: FullSequenceBatchBuilder,
    start: int,
    end: int,
    target_scale: float,
) -> np.ndarray:
    model.eval()
    output = np.empty(end - start, dtype=np.float32)
    for batch_start, batch_end in base.gru.blocks(start, end, base.EVAL_BATCH_SIZE):
        sequence = builder.make(batch_start, batch_end)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(sequence)
        output[batch_start - start:batch_end - start] = (
            prediction.float().cpu().numpy() * target_scale
        )
    return output


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    base.EPOCHS = EPOCHS
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
    base.gru.check_cache_layout(cache)
    base.gru.FOLD_NAME = FOLD_NAME
    baseline = base.gru.load_baseline(
        project_dir, labels.iloc[valid_start:valid_end]["sample_id"].to_numpy()
    )
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    fold_means, fold_scales = fit_fold_standardization(
        cache, train_end, run_dir / "fold_standardization.npz"
    )
    builder = FullSequenceBatchBuilder(
        cache, raw_means, raw_scales, fold_means, fold_scales, device
    )
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[:train_end].mean(dtype=np.float64))
    target_scale = float(target[:train_end].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_scale).astype(np.float32)
    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)
    base.gru.seed_everything(SEED)
    model = SequenceOnlyTransformer(input_size).to(device)
    check_padding_invariance(model, input_size, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base.MAX_LEARNING_RATE, weight_decay=base.WEIGHT_DECAY
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
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"resuming at epoch {start_epoch}", flush=True)

    validation_rows = labels.iloc[valid_start:valid_end]
    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        rate = base.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        train_blocks = base.gru.blocks(0, train_end, base.EFFECTIVE_BATCH_SIZE)
        np.random.default_rng(SEED + epoch).shuffle(train_blocks)
        loss_sum = 0.0
        row_count = 0
        for block_start, block_end in train_blocks:
            optimizer.zero_grad(set_to_none=True)
            block_count = block_end - block_start
            for micro_start in range(block_start, block_end, base.MICRO_BATCH_SIZE):
                micro_end = min(micro_start + base.MICRO_BATCH_SIZE, block_end)
                sequence = builder.make(micro_start, micro_end)
                batch_target = torch.as_tensor(target_scaled[micro_start:micro_end], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(sequence)
                    loss = nn.functional.mse_loss(prediction.float(), batch_target)
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
            "train_mse": loss_sum / row_count,
            "elapsed_seconds": elapsed,
        }
        logs.append(log)
        if epoch in EVALUATION_EPOCHS:
            candidate = predict(model, builder, valid_start, valid_end, target_scale).astype(np.float64)
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
                f"epoch={epoch}/{EPOCHS} loss={log['train_mse']:.7f} "
                f"cos={standalone_metrics['overall']:.7f} "
                f"blend_delta={blend_metrics['overall_delta_vs_b001']:+.7f}",
                flush=True,
            )
        else:
            print(f"epoch={epoch}/{EPOCHS} loss={log['train_mse']:.7f}", flush=True)
        pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
        base.gru.atomic_save(
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
        "model": "sequence-only Transformer",
        "static_feature_count": 0,
        "sequence_input_channels": input_size,
        "sequence_channels_include_eight_deterministic_raw-derived_channels": True,
        "seed": SEED,
        "train_months": f"0-{TRAIN_END_MONTH}",
        "validation_months": f"{VALID_START_MONTH}-{VALID_END_MONTH}",
        "training_rows": train_end,
        "validation_rows": valid_end - valid_start,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "logs": logs,
        "metrics_by_epoch": metrics_by_epoch,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
