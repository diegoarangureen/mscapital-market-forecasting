"""Test static-conditioned channel gating inside the promoted Joint-GRU."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import exp_gru_003_strong_joint as base
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import QuantileBatchPreprocessor, fit_quantile_knots


EXPERIMENT_ID = "EXP-GRU-007-GATED-JOINT-DEV"
EPOCHS = 6
EVALUATION_EPOCHS = {4, 6}


class GatedJointRegressor(base.JointRegressor):
    """Let the static market state rescale each sequence representation channel."""

    def __init__(self, sequence_input_size: int, static_input_size: int) -> None:
        super().__init__(sequence_input_size, static_input_size, use_sequence=True)
        self.sequence_gate = nn.Sequential(
            nn.Linear(base.HIDDEN_SIZE, base.HIDDEN_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, sequence: torch.Tensor | None, static: torch.Tensor) -> torch.Tensor:
        if sequence is None:
            raise ValueError("Gated Joint-GRU requires sequence input.")
        static_representation = self.static_adapter(static)
        sequence_representation = self.sequence_adapter(self.encoder(sequence))
        # 0.5--1.5 keeps a residual path while allowing static-conditioned reweighting.
        gate = 0.5 + self.sequence_gate(static_representation)
        gated_sequence = sequence_representation * gate
        combined = torch.cat([static_representation, gated_sequence], dim=1)
        return self.head(combined).squeeze(1)


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.EPOCHS = EPOCHS
    base.EVALUATION_EPOCHS = EVALUATION_EPOCHS
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
    model = GatedJointRegressor(
        input_size, preprocessor.output_dimension
    ).to(device)
    base.check_padding_invariance(model, input_size, device)
    result = base.train_candidate(
        name="joint_gated_gru319",
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
        "train_months": "0-49",
        "validation_months": "50-59",
        "fixed_epochs": EPOCHS,
        "evaluation_epochs": sorted(EVALUATION_EPOCHS),
        "feature_count": len(feature_columns),
        "gate_range": [0.5, 1.5],
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

