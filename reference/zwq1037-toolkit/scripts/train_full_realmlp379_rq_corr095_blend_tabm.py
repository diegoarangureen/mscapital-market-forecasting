"""Train full-data RQ RealMLP379 and blend it with the two-seed TabM379."""

from __future__ import annotations

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


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import exp_realmlp_005_yunsu_public_reference as reference
import exp_realmlp_006_our379_rq_reference as experiment

SOURCE_CACHE = experiment.SOURCE_CACHE
TRAIN_CACHE = experiment.OUR_CACHE
TEST_CACHE = PROJECT / "data/interim/our379_reference_test_cache"
RUN_NAME = "realmlp379_rq16_corr095_e8_fulltrain"
RUN_DIR = PROJECT / "data/interim/submissions" / RUN_NAME
TABM_PATH = PROJECT / "outputs/submissions/tabm_relative319_xs40_order20_2seed_fulltrain.csv"
REALMLP_WEIGHT = 0.32
TABM_WEIGHT = 0.68
EPOCHS = 8
SEED = reference.SEED


def ensure_test_cache(feature_names: list[str]) -> tuple[np.memmap, np.memmap]:
    TEST_CACHE.mkdir(parents=True, exist_ok=True)
    feature_path = TEST_CACHE / "features.npy"
    ids_path = TEST_CACHE / "sample_ids.npy"
    ready_path = TEST_CACHE / "READY.json"
    if ready_path.exists() and feature_path.exists() and ids_path.exists():
        features = np.load(feature_path, mmap_mode="r")
        ids = np.load(ids_path, mmap_mode="r")
        if features.shape == (len(ids), 379):
            print(f"TEST379 cache ready shape={features.shape}", flush=True)
            return features, ids

    base = np.load(SOURCE_CACHE / "test_features.npy", mmap_mode="r")
    ids = np.load(SOURCE_CACHE / "test_sample_ids.npy", mmap_mode="r")
    base_names = json.loads((SOURCE_CACHE / "feature_columns.json").read_text(encoding="utf-8"))
    groups = np.zeros(len(ids), dtype=np.int16)
    relative40, relative_names = experiment.relative_recipe.add_relative_features(
        base, groups, base_names
    )
    order_frame = pd.read_feather(
        PROJECT / "data/processed/test_order_quote_position_features.feather"
    ).sort_values("sample_id")
    if not np.array_equal(order_frame["sample_id"].to_numpy(), ids):
        raise AssertionError("Test Order20 IDs do not align")
    order20 = order_frame[experiment.ORDER_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    del order_frame
    names = [*base_names, *relative_names, *experiment.ORDER_FEATURE_COLUMNS]
    if names != feature_names:
        raise AssertionError("Train and test 379 feature names differ")
    output = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(len(ids), 379)
    )
    block = 32768
    for start in range(0, len(ids), block):
        stop = min(start + block, len(ids))
        output[start:stop, :319] = base[start:stop]
        output[start:stop, 319:359] = relative40[start:stop]
        output[start:stop, 359:] = order20[start:stop]
    output.flush()
    ids_output = np.lib.format.open_memmap(
        ids_path, mode="w+", dtype=np.int64, shape=(len(ids),)
    )
    ids_output[:] = ids
    ids_output.flush()
    ready_path.write_text(
        json.dumps({"rows": len(ids), "features": 379}, indent=2), encoding="utf-8"
    )
    print(f"TEST379 cache completed shape={output.shape}", flush=True)
    return np.load(feature_path, mmap_mode="r"), np.load(ids_path, mmap_mode="r")


def fill_missing(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train[~np.isfinite(train)] = np.nan
    test[~np.isfinite(test)] = np.nan
    medians = np.nanmedian(train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0).astype(np.float32)
    train_missing = np.where(~np.isfinite(train))
    test_missing = np.where(~np.isfinite(test))
    train[train_missing] = medians[train_missing[1]]
    test[test_missing] = medians[test_missing[1]]
    return train, test


def low_memory_select(
    sample_ids: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str], dict]:
    """Run Yunsu correlation pruning on virtual [sample_id, features] chunks."""
    rows = len(features)
    columns = features.shape[1] + 1
    sum_x = np.zeros(columns, dtype=np.float64)
    sum_x2 = np.zeros(columns, dtype=np.float64)
    sum_xy = np.zeros(columns, dtype=np.float64)
    cross = np.zeros((columns, columns), dtype=np.float64)
    sum_y = 0.0
    sum_y2 = 0.0
    block = 8192
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        x = np.empty((stop - start, columns), dtype=np.float64)
        x[:, 0] = sample_ids[start:stop]
        x[:, 1:] = features[start:stop]
        y = np.asarray(target[start:stop], dtype=np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += np.einsum("ij,ij->j", x, x)
        sum_xy += x.T @ y
        cross += x.T @ x
        sum_y += float(y.sum())
        sum_y2 += float(np.dot(y, y))
    variance_x = np.maximum(sum_x2 - sum_x * sum_x / rows, 0.0)
    variance_y = max(sum_y2 - sum_y * sum_y / rows, 0.0)
    covariance_xy = sum_xy - sum_x * sum_y / rows
    target_corr = np.abs(covariance_xy / (np.sqrt(variance_x * variance_y) + 1e-30))
    covariance_x = cross - np.outer(sum_x, sum_x) / rows
    correlation = np.abs(
        covariance_x / (np.sqrt(variance_x[:, None] * variance_x[None, :]) + 1e-30)
    )
    upper_i, upper_j = np.triu_indices(columns, k=1)
    pair_values = correlation[upper_i, upper_j]
    high = np.flatnonzero(pair_values >= 0.95)
    order = high[np.argsort(pair_values[high])[::-1]]
    dropped: set[int] = set()
    for position in order:
        left = int(upper_i[position])
        right = int(upper_j[position])
        if left in dropped or right in dropped:
            continue
        dropped.add(right if target_corr[left] >= target_corr[right] else left)
    for index in range(columns):
        if variance_x[index] == 0.0 or target_corr[index] < 0.0001:
            dropped.add(index)
    keep = np.asarray([index - 1 for index in range(1, columns) if index not in dropped])
    kept_names = [names[index] for index in keep]
    details = {
        "input_features": len(names),
        "kept_features": len(kept_names),
        "sample_id_dropped": 0 in dropped,
        "dropped_features": [names[index - 1] for index in sorted(dropped) if index > 0],
    }
    return keep, kept_names, details


def prepare_in_place(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    """Yunsu binning/category/RSSC preprocessing without duplicate input copies."""
    unique_counts = []
    for column in range(train.shape[1]):
        count = int(np.unique(train[:, column]).size)
        if count > 100:
            train[:, column], test[:, column] = reference.cut_with_train_quantiles(
                train[:, column], test[:, column]
            )
            count = int(np.unique(train[:, column]).size)
        unique_counts.append(count)
    categorical_indices = [i for i, count in enumerate(unique_counts) if count <= reference.ONE_HOT_MAX]
    numerical_indices = [i for i, count in enumerate(unique_counts) if count > reference.ONE_HOT_MAX]
    train_cat = np.empty((len(train), len(categorical_indices)), dtype=np.float32)
    test_cat = np.empty((len(test), len(categorical_indices)), dtype=np.float32)
    cat_dims = []
    for output_column, source_column in enumerate(categorical_indices):
        categories, first = np.unique(train[:, source_column], return_index=True)
        categories = categories[np.argsort(first)]
        mapping = {float(value): index for index, value in enumerate(categories)}
        train_cat[:, output_column] = np.fromiter(
            (mapping[float(value)] for value in train[:, source_column]),
            dtype=np.float32,
            count=len(train),
        )
        test_cat[:, output_column] = np.fromiter(
            (mapping.get(float(value), 0) for value in test[:, source_column]),
            dtype=np.float32,
            count=len(test),
        )
        cat_dims.append(len(categories))
    train_num = train[:, numerical_indices].astype(np.float32, copy=True)
    test_num = test[:, numerical_indices].astype(np.float32, copy=True)
    median = np.median(train_num, axis=0)
    q25 = np.quantile(train_num, 0.25, axis=0)
    q75 = np.quantile(train_num, 0.75, axis=0)
    difference = q75 - q25
    zero = difference == 0.0
    if np.any(zero):
        value_range = np.max(train_num, axis=0) - np.min(train_num, axis=0)
        difference[zero] = 0.5 * value_range[zero]
    factors = 1.0 / (difference + 1e-30)
    factors[difference == 0.0] = 0.0
    train_scaled = factors[None] * (train_num - median[None])
    test_scaled = factors[None] * (test_num - median[None])
    train_num = (train_scaled / np.sqrt(1.0 + (train_scaled / 3.0) ** 2)).astype(np.float32)
    test_num = (test_scaled / np.sqrt(1.0 + (test_scaled / 3.0) ** 2)).astype(np.float32)
    metadata = {
        "categorical_features": len(categorical_indices),
        "numerical_features": len(numerical_indices),
        "cat_dims": cat_dims,
    }
    return train_num, test_num, train_cat, test_cat, cat_dims, metadata


def main() -> None:
    torch.set_num_threads(2)
    reference.set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for directory in ("outputs/models", "outputs/predictions", "outputs/submissions", "outputs/submission_metadata"):
        (PROJECT / directory).mkdir(parents=True, exist_ok=True)

    train_features, target_map, feature_names = experiment.ensure_our379_cache()
    test_features, test_ids = ensure_test_cache(feature_names)
    train_ids = np.load(SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    target = np.asarray(target_map, dtype=np.float32).copy()
    raw_train = np.array(train_features, dtype=np.float32, copy=True)
    raw_test = np.array(test_features, dtype=np.float32, copy=True)
    raw_train, raw_test = fill_missing(raw_train, raw_test)
    keep, kept_names, selection = low_memory_select(train_ids, raw_train, target, feature_names)
    selected_train = raw_train[:, keep].copy()
    selected_test = raw_test[:, keep].copy()
    del raw_train, raw_test
    gc.collect()
    print(f"selected_features={len(kept_names)}/379", flush=True)
    train_num, test_num, train_cat, test_cat, cat_dims, preprocessing = prepare_in_place(
        selected_train, selected_test
    )
    del selected_train, selected_test
    gc.collect()

    rounded_target = np.round(target, 4).astype(np.float32)
    rq_encoder = reference.RQKMeansEncoder().fit(rounded_target)
    rq_codes = rq_encoder.encode(rounded_target)
    device = torch.device("cuda")
    train_num_gpu = torch.from_numpy(train_num).to(device)
    train_cat_gpu = torch.from_numpy(train_cat).to(device)
    target_gpu = torch.from_numpy(rounded_target).to(device)
    codes_gpu = torch.from_numpy(rq_codes).to(device)
    del train_num, train_cat, rq_codes
    gc.collect()

    model = reference.RealMLPRQ(train_num_gpu.shape[1], cat_dims).to(device)
    optimizer = torch.optim.AdamW(reference.parameter_groups(model), betas=(0.9, 0.98))
    ema = reference.EMA(model, decay=0.998)
    batch_size = reference.TRAIN_BATCH_SIZE
    steps_per_epoch = (len(target) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * EPOCHS
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    logs = []
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        permutation = torch.randperm(len(target), generator=generator)
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size].to(device)
            global_step = (epoch - 1) * steps_per_epoch + start // batch_size
            progress = min(global_step / total_steps, 1.0)
            initial_rates = [
                reference.LEARNING_RATE * 20.0,
                reference.LEARNING_RATE * 0.093,
                reference.LEARNING_RATE,
                reference.LEARNING_RATE,
                reference.LEARNING_RATE * 0.1,
            ]
            for group, initial_rate in zip(optimizer.param_groups, initial_rates):
                group["lr"] = reference.flat_anneal(initial_rate, progress)
            noisy_target = target_gpu[indices] + torch.randn_like(target_gpu[indices]) * (
                0.005 * (1.0 - progress)
            )
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = model(
                train_num_gpu[indices], train_cat_gpu[indices], return_codes=True
            )
            loss, cosine, mse, rq_loss = reference.compute_loss(
                prediction,
                noisy_target,
                logits,
                codes_gpu[indices],
                reference.flat_anneal(0.1, progress),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            sums += [float(loss.detach()), float(cosine.detach()), float(mse.detach()), float(rq_loss.detach())]
            batches += 1
        row = {
            "epoch": epoch,
            "loss": float(sums[0] / batches),
            "train_cosine": float(sums[1] / batches),
            "mse": float(sums[2] / batches),
            "rq_loss": float(sums[3] / batches),
            "seconds": time.time() - started,
        }
        logs.append(row)
        print("EPOCH " + json.dumps(row), flush=True)

    ema.apply()
    model_path = PROJECT / "outputs/models" / f"{RUN_NAME}.pt"
    torch.save(model.state_dict(), model_path)
    del train_num_gpu, train_cat_gpu, target_gpu, codes_gpu
    torch.cuda.empty_cache()
    gc.collect()

    test_num_gpu = torch.from_numpy(test_num).to(device)
    test_cat_gpu = torch.from_numpy(test_cat).to(device)
    del test_num, test_cat
    model.eval()
    prediction_parts = []
    with torch.no_grad():
        for start in range(0, len(test_ids), reference.EVAL_BATCH_SIZE):
            stop = min(start + reference.EVAL_BATCH_SIZE, len(test_ids))
            prediction = model(test_num_gpu[start:stop], test_cat_gpu[start:stop]).squeeze(-1)
            prediction_parts.append(prediction.cpu().numpy())
    realmlp_prediction = np.concatenate(prediction_parts).astype(np.float64)
    if len(realmlp_prediction) != len(test_ids) or not np.isfinite(realmlp_prediction).all():
        raise AssertionError("Invalid RealMLP test prediction")

    realmlp_path = PROJECT / "outputs/submissions" / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": realmlp_prediction}).to_csv(
        realmlp_path, index=False
    )
    tabm_frame = pd.read_csv(TABM_PATH).sort_values("sample_id")
    if not np.array_equal(tabm_frame["sample_id"].to_numpy(), test_ids):
        raise AssertionError("TabM submission IDs do not align")
    tabm_prediction = tabm_frame["prediction"].to_numpy(dtype=np.float64)
    realmlp_unit = realmlp_prediction / np.linalg.norm(realmlp_prediction)
    tabm_unit = tabm_prediction / np.linalg.norm(tabm_prediction)
    blended_unit = REALMLP_WEIGHT * realmlp_unit + TABM_WEIGHT * tabm_unit
    blended_prediction = blended_unit * np.linalg.norm(tabm_prediction)
    blend_name = "realmlp379_rq16_corr095_e8_w32_tabm379_2seed_w68"
    blend_path = PROJECT / "outputs/submissions" / f"{blend_name}.csv"
    pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": blended_prediction}).to_csv(
        blend_path, index=False
    )
    prediction_path = PROJECT / "outputs/predictions" / f"{RUN_NAME}_test.feather"
    pd.DataFrame({
        "sample_id": np.asarray(test_ids),
        "realmlp_prediction": realmlp_prediction,
        "tabm_prediction": tabm_prediction,
        "blended_prediction": blended_prediction,
    }).to_feather(prediction_path)
    metadata = {
        "run_name": RUN_NAME,
        "model": "Yunsu RQ RealMLP, 16 members, EMA",
        "features": "our Relative319 + XS40 + Order20",
        "input_features": 379,
        "kept_features": len(kept_names),
        "kept_feature_names": kept_names,
        "selection": selection,
        "preprocessing": preprocessing,
        "epochs": EPOCHS,
        "seed": SEED,
        "validation_centered_cosine": 0.15348300645832985,
        "validation_raw_cosine": 0.15466833910919217,
        "validation_tabm_raw_cosine": 0.15986343921046062,
        "validation_blend_raw_cosine": 0.16134594147361084,
        "blend_weights": {"realmlp": REALMLP_WEIGHT, "tabm_two_seed": TABM_WEIGHT},
        "realmlp_prediction_mean": float(realmlp_prediction.mean()),
        "realmlp_prediction_std": float(realmlp_prediction.std()),
        "blend_prediction_mean": float(blended_prediction.mean()),
        "blend_prediction_std": float(blended_prediction.std()),
        "logs": logs,
        "model_path": str(model_path),
        "realmlp_submission_path": str(realmlp_path),
        "blend_submission_path": str(blend_path),
        "prediction_path": str(prediction_path),
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path = PROJECT / "outputs/submission_metadata" / f"{blend_name}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


