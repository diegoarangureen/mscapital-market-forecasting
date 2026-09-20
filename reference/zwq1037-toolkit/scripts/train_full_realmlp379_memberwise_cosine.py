"""Train the validated four-member RealMLP on all months and predict Kaggle test."""

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "data/interim/kaggle_kernels/relative319_xs_tabm_notebook"))
import exp_realmlp_003_yunsu_memberwise_cosine as dev
import run_extracted as recipe
from build_order_quote_position_features import FEATURE_COLUMNS

CACHE = PROJECT / "data/interim/kaggle_relative319_dev"
RUN_NAME = "realmlp379_memberwise_cosine_seed42_e3_fulltrain"
RUN_DIR = PROJECT / "data/interim/submissions" / RUN_NAME
EPOCHS = 3
BATCH_SIZE = 1024
SEED = 42


def build_gpu_features(base, xs, order, median, scale):
    # 分块预处理后常驻显存，避免训练时反复随机读取磁盘。
    # Preprocess in chunks and keep the matrix on GPU during training.
    output = torch.empty((len(base), 379), dtype=torch.float16, device="cuda")
    chunk_rows = 16384
    for start in range(0, len(base), chunk_rows):
        stop = min(start + chunk_rows, len(base))
        values = np.concatenate((base[start:stop], xs[start:stop], order[start:stop]), axis=1)
        values = np.where(np.isfinite(values), values, median)
        values = np.clip((values - median) / scale, -10, 10).astype(np.float16)
        output[start:stop].copy_(torch.from_numpy(values))
    return output


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT / "outputs/models").mkdir(parents=True, exist_ok=True)
    (PROJECT / "outputs/predictions").mkdir(parents=True, exist_ok=True)
    (PROJECT / "outputs/submissions").mkdir(parents=True, exist_ok=True)
    (PROJECT / "outputs/submission_metadata").mkdir(parents=True, exist_ok=True)

    train_base = np.load(CACHE / "features.npy", mmap_mode="r")
    train_ids = np.load(CACHE / "sample_ids.npy", mmap_mode="r")
    train_months = np.load(CACHE / "months.npy", mmap_mode="r")
    target = np.load(CACHE / "targets.npy", mmap_mode="r")
    columns = json.loads((CACHE / "feature_columns.json").read_text(encoding="utf-8"))
    train_xs, xs_names = recipe.add_relative_features(train_base, train_months, columns)
    train_order_frame = pd.read_feather(
        PROJECT / "data/processed/train_order_quote_position_features.feather"
    ).sort_values("sample_id")
    if not np.array_equal(train_order_frame.sample_id.to_numpy(), train_ids):
        raise AssertionError("Train order20 IDs do not align")
    train_order = train_order_frame[FEATURE_COLUMNS].to_numpy(np.float32)
    del train_order_frame

    test_base = np.load(CACHE / "test_features.npy", mmap_mode="r")
    test_ids = np.load(CACHE / "test_sample_ids.npy", mmap_mode="r")
    test_groups = np.zeros(len(test_ids), dtype=np.int16)
    test_xs, test_xs_names = recipe.add_relative_features(test_base, test_groups, columns)
    if test_xs_names != xs_names:
        raise AssertionError("Train and test XS feature names differ")
    test_order_frame = pd.read_feather(
        PROJECT / "data/processed/test_order_quote_position_features.feather"
    ).sort_values("sample_id")
    if not np.array_equal(test_order_frame.sample_id.to_numpy(), test_ids):
        raise AssertionError("Test order20 IDs do not align")
    test_order = test_order_frame[FEATURE_COLUMNS].to_numpy(np.float32)
    del test_order_frame

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(train_base), size=50000, replace=False)
    sample = np.concatenate(
        (train_base[sample_idx], train_xs[sample_idx], train_order[sample_idx]), axis=1
    )
    sample[~np.isfinite(sample)] = np.nan
    median = np.nan_to_num(np.nanmedian(sample, axis=0), nan=0).astype(np.float32)
    q25 = np.nanpercentile(sample, 25, axis=0)
    q75 = np.nanpercentile(sample, 75, axis=0)
    scale = np.maximum(np.nan_to_num(q75 - q25, nan=1), 1e-4).astype(np.float32)
    np.savez_compressed(
        RUN_DIR / "robust_preprocessing.npz",
        median=median,
        scale=scale,
        feature_count=np.asarray([379]),
    )
    del sample
    gc.collect()

    target_array = np.asarray(target, dtype=np.float32).copy()
    target_mean = float(target_array.mean(dtype=np.float64))
    target_std = float(target_array.std(dtype=np.float64))
    gpu_target = torch.from_numpy(target_array).to("cuda")
    gpu_train = build_gpu_features(train_base, train_xs, train_order, median, scale)
    del train_xs, train_order
    gc.collect()

    model = dev.NumericYunsuMLP(379).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    logs = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        shuffled = rng.permutation(len(train_base))
        losses = []
        for start in range(0, len(shuffled), BATCH_SIZE):
            batch_idx = shuffled[start:start + BATCH_SIZE]
            gpu_idx = torch.as_tensor(batch_idx, dtype=torch.long, device="cuda")
            values = gpu_train[gpu_idx]
            scaled_y = (gpu_target[gpu_idx] - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                member_prediction = model(values).float()
                repeated_target = scaled_y[:, None].expand_as(member_prediction)
                sample_weight = torch.where(
                    gpu_target[gpu_idx].abs() > 0.001, 0.5, 1.0
                )[:, None]
                mse_loss = (
                    sample_weight * (member_prediction - repeated_target).square()
                ).mean()
                pred_flat = member_prediction.flatten()
                target_flat = repeated_target.flatten()
                cosine_loss = 1.0 - F.cosine_similarity(
                    pred_flat - pred_flat.mean(),
                    target_flat - target_flat.mean(),
                    dim=0,
                )
                loss = mse_loss + 0.01 * cosine_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": time.perf_counter() - started,
        }
        logs.append(row)
        torch.save(model.state_dict(), RUN_DIR / f"model_epoch{epoch}.pt")
        print(json.dumps(row), flush=True)

    del gpu_train, gpu_target
    torch.cuda.empty_cache()
    gc.collect()
    gpu_test = build_gpu_features(test_base, test_xs, test_order, median, scale)
    del test_xs, test_order
    gc.collect()

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(test_ids), BATCH_SIZE):
            with torch.autocast("cuda", dtype=torch.float16):
                members = model(gpu_test[start:start + BATCH_SIZE])
            pieces.append(members.mean(dim=1).float().cpu().numpy())
    prediction = np.concatenate(pieces).astype(np.float64) * target_std + target_mean
    if len(prediction) != len(test_ids) or not np.isfinite(prediction).all():
        raise AssertionError("Invalid test prediction")
    prediction_path = PROJECT / "outputs/predictions" / f"{RUN_NAME}_test.feather"
    submission_path = PROJECT / "outputs/submissions" / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": prediction}).to_feather(
        prediction_path
    )
    pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": prediction}).to_csv(
        submission_path, index=False
    )
    model_path = PROJECT / "outputs/models" / f"{RUN_NAME}.pt"
    torch.save(model.state_dict(), model_path)
    metadata = {
        "run_name": RUN_NAME,
        "model": "4-member numeric Yunsu-style PBLD RealMLP",
        "features": "Relative319 + XS40 + order20",
        "feature_count": 379,
        "loss": "member-wise weighted MSE + 0.01 centered cosine",
        "seed": SEED,
        "epochs": EPOCHS,
        "train_months": "0-70",
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "validation": {
            "train049_valid5059_standalone": 0.13713810258461037,
            "train059_valid6270_ex66_standalone": 0.1448919632456695,
        },
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "logs": logs,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "submission_path": str(submission_path),
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path = PROJECT / "outputs/submission_metadata" / f"{RUN_NAME}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

