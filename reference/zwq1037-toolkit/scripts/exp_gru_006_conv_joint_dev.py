"""Test one kernel-5 residual convolution before the promoted joint GRU."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import exp_gru_003_strong_joint as base
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import QuantileBatchPreprocessor, fit_quantile_knots


EXPERIMENT_ID = "EXP-GRU-006-CONV-JOINT-DEV"
EPOCHS = 6
EVALUATION_EPOCHS = {4, 6}
KERNEL_SIZE = 5


class ConvSequenceEncoder(nn.Module):
    """Add a mask-aware local convolution while preserving the GRU pooling design."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, base.HIDDEN_SIZE),
            nn.SiLU(),
            nn.LayerNorm(base.HIDDEN_SIZE),
        )
        self.local_convolution = nn.Conv1d(
            base.HIDDEN_SIZE,
            base.HIDDEN_SIZE,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
        )
        self.local_norm = nn.LayerNorm(base.HIDDEN_SIZE)
        self.gru = nn.GRU(
            base.HIDDEN_SIZE, base.HIDDEN_SIZE, num_layers=1, batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(base.HIDDEN_SIZE, base.HIDDEN_SIZE // 2),
            nn.Tanh(),
            nn.Linear(base.HIDDEN_SIZE // 2, 1),
        )

    @property
    def output_size(self) -> int:
        return base.HIDDEN_SIZE * 4

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence, valid_prefix, lengths = base.left_align_valid_events(sequence)
        mask = valid_prefix[:, :, None]
        projected = self.input_projection(sequence) * mask
        local = self.local_convolution(projected.transpose(1, 2)).transpose(1, 2)
        projected = self.local_norm(projected + torch.nn.functional.silu(local)) * mask

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
        full_mean = (recurrent * mask).sum(dim=1) / lengths[:, None]

        positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        recent_start = torch.clamp(lengths - base.RECENT_EVENT_COUNT, min=0)
        recent_mask = (positions >= recent_start[:, None]) & valid_prefix
        recent_count = recent_mask.sum(dim=1).clamp_min(1)
        recent_mean = (
            (recurrent * recent_mask[:, :, None]).sum(dim=1) / recent_count[:, None]
        )

        attention_logits = self.attention(recurrent).squeeze(2)
        attention_logits = attention_logits.masked_fill(~valid_prefix, -torch.inf)
        attention_weights = torch.softmax(attention_logits.float(), dim=1).to(recurrent.dtype)
        attention_pool = (recurrent * attention_weights[:, :, None]).sum(dim=1)
        return torch.cat([hidden[-1], full_mean, recent_mean, attention_pool], dim=1)


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EPOCHS = EPOCHS
    base.EVALUATION_EPOCHS = EVALUATION_EPOCHS
    base.StrongSequenceEncoder = ConvSequenceEncoder
    device = torch.device("cuda")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = (
        project_dir / "data" / "interim" / "sequence_experiments" / EXPERIMENT_ID
    )
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
    train_end = int(np.searchsorted(months, base.TRAIN_END_MONTH + 1, side="left"))
    valid_start = int(np.searchsorted(months, base.VALID_START_MONTH, side="left"))
    valid_end = int(np.searchsorted(months, base.VALID_END_MONTH + 1, side="left"))
    validation_rows = labels.iloc[valid_start:valid_end].copy()
    baseline = base.load_baseline(
        project_dir, validation_rows["sample_id"].to_numpy()
    )

    base.check_cache_layout(cache)
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

    static_features, feature_columns = base.load_relative319(project_dir, labels)
    preprocessing_path = run_dir / "quantile_preprocessing.npz"
    train_indices = np.arange(train_end, dtype=np.int64)
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
    else:
        knots, medians, missing_columns = fit_quantile_knots(
            static_features, train_indices, base.SEED
        )
        np.savez_compressed(
            preprocessing_path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray(feature_columns),
        )
    preprocessor = QuantileBatchPreprocessor(knots, medians, missing_columns, device)

    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)
    base.seed_everything(base.SEED)
    model = base.JointRegressor(
        input_size, preprocessor.output_dimension, use_sequence=True
    ).to(device)
    base.check_padding_invariance(model, input_size, device)
    result = base.train_candidate(
        name="joint_conv_gru319",
        model=model,
        builder=builder,
        target_scaled=target_scaled,
        target_scale=target_scale,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        validation_rows=validation_rows,
        baseline=baseline,
        run_dir=run_dir,
        static_features=static_features,
        static_preprocessor=preprocessor,
        device=device,
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "seed": base.SEED,
        "train_months": f"0-{base.TRAIN_END_MONTH}",
        "validation_months": f"{base.VALID_START_MONTH}-{base.VALID_END_MONTH}",
        "change": "kernel-5 full-channel residual convolution before the existing GRU",
        "kernel_size": KERNEL_SIZE,
        "feature_count": len(feature_columns),
        "model": result,
    }
    (run_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()


