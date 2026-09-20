"""Train the validated Conv Hybrid Joint-Transformer on all months."""

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

import exp_transformer_010_conv_hybrid_dev as transformer
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from train_full_gru_embedding_candidate import atomic_save, fit_preprocessor, load_test_relative319


RUN_NAME = "joint_transformer_conv_hybrid_epoch5_fulltrain"
SEED = 42
EPOCHS = 5


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    transformer.EPOCHS = EPOCHS
    device = torch.device("cuda")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "submissions" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    test_cache_dir = project_dir / "data" / "processed" / "test_sequence_cache_v1"
    train_cache = np.load(train_cache_dir / "sequences.npy", mmap_mode="r")
    test_cache = np.load(test_cache_dir / "sequences.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    if train_cache.shape != (len(labels), 14, 200):
        raise AssertionError("Train sequence cache and labels differ.")
    if set(labels["month"].unique()) != set(range(71)):
        raise AssertionError("Full training data must contain months 0-70.")
    transformer.gru.check_cache_layout(train_cache)
    raw_means, raw_scales = load_raw_statistics(train_cache_dir)
    fold_means, fold_scales = fit_fold_standardization(
        train_cache, len(labels), run_dir / "full_standardization.npz"
    )
    train_builder = FullSequenceBatchBuilder(
        train_cache, raw_means, raw_scales, fold_means, fold_scales, device
    )
    train_features, feature_columns = transformer.gru.load_relative319(project_dir, labels)
    preprocessor = fit_preprocessor(
        train_features,
        feature_columns,
        SEED,
        run_dir / "quantile_preprocessing.npz",
        device,
    )
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target.mean(dtype=np.float64))
    target_scale = float(target.std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_scale).astype(np.float32)
    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)
    transformer.gru.seed_everything(SEED)
    model = transformer.JointTransformerRegressor(
        input_size, preprocessor.output_dimension
    ).to(device)
    transformer.check_padding_invariance(
        model, input_size, preprocessor.output_dimension, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=transformer.MAX_LEARNING_RATE,
        weight_decay=transformer.WEIGHT_DECAY,
    )
    checkpoint_path = run_dir / "training_checkpoint.pt"
    logs: list[dict] = []
    start_epoch = 1
    elapsed_before = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logs = list(checkpoint["logs"])
        start_epoch = int(checkpoint["next_epoch"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"resuming at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        rate = transformer.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        train_blocks = transformer.gru.blocks(
            0, len(labels), transformer.EFFECTIVE_BATCH_SIZE
        )
        np.random.default_rng(SEED + epoch).shuffle(train_blocks)
        loss_sum = 0.0
        row_count = 0
        for block_start, block_end in train_blocks:
            optimizer.zero_grad(set_to_none=True)
            block_count = block_end - block_start
            for micro_start in range(
                block_start, block_end, transformer.MICRO_BATCH_SIZE
            ):
                micro_end = min(
                    micro_start + transformer.MICRO_BATCH_SIZE, block_end
                )
                sequence = train_builder.make(micro_start, micro_end)
                static = preprocessor.transform(train_features[micro_start:micro_end])
                batch_target = torch.as_tensor(
                    target_scaled[micro_start:micro_end], device=device
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(sequence, static)
                    smooth_l1 = nn.functional.smooth_l1_loss(prediction.float(), batch_target)
                    cosine = transformer.centered_cosine_loss(prediction, batch_target)
                    loss = transformer.SMOOTH_L1_WEIGHT * smooth_l1 + transformer.COSINE_WEIGHT * cosine
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
        pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "elapsed_seconds": elapsed,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_path,
        )
        print(
            f"full Joint-Transformer epoch={epoch}/{EPOCHS} "
            f"lr={rate:.7f} loss={log['train_loss']:.7f}",
            flush=True,
        )

    model_path = project_dir / "outputs" / "models" / f"{RUN_NAME}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(
        {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "static_input_size": preprocessor.output_dimension,
            "feature_columns": feature_columns,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "seed": SEED,
            "epochs": EPOCHS,
        },
        model_path,
    )
    del train_features, target_scaled, train_builder, train_cache
    gc.collect()

    test_features, template, test_columns = load_test_relative319(project_dir)
    if test_columns != feature_columns:
        raise AssertionError("Train and test relative319 schemas differ.")
    if test_cache.shape != (len(template), 14, 200):
        raise AssertionError("Test sequence cache and template differ.")
    transformer.gru.check_cache_layout(test_cache)
    test_builder = FullSequenceBatchBuilder(
        test_cache, raw_means, raw_scales, fold_means, fold_scales, device
    )
    prediction = transformer.predict(
        model,
        test_builder,
        test_features,
        preprocessor,
        0,
        len(template),
        target_scale,
    ).astype(np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError("Joint-Transformer prediction contains invalid values.")

    prediction_dir = project_dir / "outputs" / "predictions"
    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    for directory in (prediction_dir, submission_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{RUN_NAME}_test.feather"
    submission_path = submission_dir / f"{RUN_NAME}.csv"
    pd.DataFrame(
        {"sample_id": template["sample_id"].to_numpy(), "prediction": prediction}
    ).to_feather(prediction_path)
    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    submission.to_csv(submission_path, index=False)
    metadata = {
        "run_name": RUN_NAME,
        "status": "complete",
        "submission_status": "prepared_not_uploaded",
        "model": "standalone Conv Hybrid Joint-Transformer",
        "loss": "0.35 SmoothL1 + 0.65 centered cosine",
        "training_months": "0-70",
        "train_rows": len(labels),
        "test_rows": len(template),
        "static_feature_count": len(feature_columns),
        "sequence_input_channels": input_size,
        "seed": SEED,
        "epochs": EPOCHS,
        "architecture": {
            "d_model": transformer.D_MODEL,
            "heads": transformer.NUM_HEADS,
            "layers": transformer.NUM_LAYERS,
            "feedforward_size": transformer.FEEDFORWARD_SIZE,
            "temporal_convolutions": [5, 3],
        },
        "training_logs": logs,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "submission_path": str(submission_path),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(ddof=0)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    }
    (metadata_dir / f"{RUN_NAME}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()



