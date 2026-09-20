"""Train three temporal cutoffs of RQ RealMLP379 with shared preprocessing."""
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
import train_full_realmlp379_rq_corr095_blend_tabm as base

reference = base.reference
experiment = base.experiment
RUN_NAME = "realmlp379_rq16_temporal3fold_e8"
RUN_DIR = PROJECT / "data/interim/submissions" / RUN_NAME
MODEL_DIR = PROJECT / "outputs/models" / RUN_NAME
PRED_DIR = PROJECT / "outputs/predictions" / RUN_NAME
OUT_DIR = PROJECT / "outputs/submissions"
META_PATH = PROJECT / "outputs/submission_metadata" / f"{RUN_NAME}.json"
FOLD_ENDS = (52, 57, 62)
EPOCHS = 8
SEED = reference.SEED


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def train_fold(
    fold_end, train_indices, train_num_gpu, train_cat_gpu,
    target_gpu, codes_gpu, cat_dims,
):
    reference.set_seed(SEED)
    model = reference.RealMLPRQ(train_num_gpu.shape[1], cat_dims).to("cuda")
    optimizer = torch.optim.AdamW(reference.parameter_groups(model), betas=(0.9, 0.98))
    ema = reference.EMA(model, decay=0.998)
    batch_size = reference.TRAIN_BATCH_SIZE
    steps_per_epoch = (len(train_indices) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * EPOCHS
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train_indices_cpu = torch.as_tensor(train_indices, dtype=torch.int64)
    logs = []
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(train_indices), generator=generator)
        permutation = train_indices_cpu[order]
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size].to("cuda")
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
                prediction, noisy_target, logits, codes_gpu[indices],
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
        print(f"fold_end={fold_end} " + json.dumps(row), flush=True)
    ema.apply()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model_path = MODEL_DIR / f"fold_end{fold_end}_seed{SEED}.pt"
    torch.save(state, model_path)
    del model, optimizer, ema
    torch.cuda.empty_cache()
    gc.collect()
    return state, logs, model_path


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    for directory in (RUN_DIR, MODEL_DIR, PRED_DIR, OUT_DIR, META_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)

    train_features, target_map, feature_names = experiment.ensure_our379_cache()
    test_features, test_ids = base.ensure_test_cache(feature_names)
    train_ids = np.load(base.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(base.SOURCE_CACHE / "months.npy", mmap_mode="r")
    target = np.asarray(target_map, dtype=np.float32).copy()
    raw_train = np.array(train_features, dtype=np.float32, copy=True)
    raw_test = np.array(test_features, dtype=np.float32, copy=True)
    raw_train, raw_test = base.fill_missing(raw_train, raw_test)
    keep, kept_names, selection = base.low_memory_select(
        train_ids, raw_train, target, feature_names
    )
    selected_train = raw_train[:, keep].copy()
    selected_test = raw_test[:, keep].copy()
    del raw_train, raw_test
    gc.collect()
    print(f"selected_features={len(kept_names)}/379", flush=True)
    train_num, test_num, train_cat, test_cat, cat_dims, preprocessing = base.prepare_in_place(
        selected_train, selected_test
    )
    del selected_train, selected_test
    gc.collect()

    rounded_target = np.round(target, 4).astype(np.float32)
    rq_encoder = reference.RQKMeansEncoder().fit(rounded_target)
    rq_codes = rq_encoder.encode(rounded_target)
    train_num_gpu = torch.from_numpy(train_num).to("cuda")
    train_cat_gpu = torch.from_numpy(train_cat).to("cuda")
    target_gpu = torch.from_numpy(rounded_target).to("cuda")
    codes_gpu = torch.from_numpy(rq_codes).to("cuda")
    del train_num, train_cat, rq_codes
    gc.collect()

    states = []
    fold_results = []
    for fold_end in FOLD_ENDS:
        train_indices = np.flatnonzero(months <= fold_end)
        state, logs, model_path = train_fold(
            fold_end, train_indices, train_num_gpu, train_cat_gpu,
            target_gpu, codes_gpu, cat_dims,
        )
        states.append((fold_end, state))
        fold_results.append({
            "fold_end": fold_end,
            "train_rows": int(len(train_indices)),
            "model_path": str(model_path),
            "logs": logs,
        })
    del train_num_gpu, train_cat_gpu, target_gpu, codes_gpu
    torch.cuda.empty_cache()
    gc.collect()

    test_num_gpu = torch.from_numpy(test_num).to("cuda")
    test_cat_gpu = torch.from_numpy(test_cat).to("cuda")
    del test_num, test_cat
    predictions = []
    for fold_end, state in states:
        model = reference.RealMLPRQ(test_num_gpu.shape[1], cat_dims).to("cuda")
        model.load_state_dict(state)
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(test_ids), reference.EVAL_BATCH_SIZE):
                stop = min(start + reference.EVAL_BATCH_SIZE, len(test_ids))
                part = model(test_num_gpu[start:stop], test_cat_gpu[start:stop]).squeeze(-1)
                parts.append(part.cpu().numpy())
        prediction = np.concatenate(parts).astype(np.float64)
        if len(prediction) != len(test_ids) or not np.isfinite(prediction).all():
            raise AssertionError(f"Invalid fold_end={fold_end} prediction")
        path = OUT_DIR / f"{RUN_NAME}_fold_end{fold_end}.csv"
        pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": prediction}).to_csv(path, index=False)
        pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": prediction}).to_feather(
            PRED_DIR / f"fold_end{fold_end}.feather"
        )
        predictions.append(prediction)
        for row in fold_results:
            if row["fold_end"] == fold_end:
                row["prediction_path"] = str(path)
                row["prediction_mean"] = float(prediction.mean())
                row["prediction_std"] = float(prediction.std())
        del model, state
        torch.cuda.empty_cache()
        gc.collect()

    ensemble = np.mean([unit(prediction) for prediction in predictions], axis=0)
    submission_path = OUT_DIR / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": np.asarray(test_ids), "prediction": ensemble}).to_csv(
        submission_path, index=False
    )
    report = {
        "run_name": RUN_NAME,
        "model": "Yunsu RQ RealMLP, 16 members, EMA",
        "features": "Relative319 + XS40 + Order20",
        "preprocessing_scope": "shared full labeled data; only training row cutoff varies",
        "fold_ends": list(FOLD_ENDS),
        "seed": SEED,
        "epochs_per_model": EPOCHS,
        "kept_features": len(kept_names),
        "selection": selection,
        "preprocessing": preprocessing,
        "folds": fold_results,
        "pairwise_prediction_correlation": np.corrcoef(np.column_stack(predictions), rowvar=False).tolist(),
        "submission_path": str(submission_path),
        "prediction_std": float(ensemble.std()),
        "submission_status": "prepared_not_submitted",
    }
    META_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps({
        "submission_path": str(submission_path),
        "prediction_std": report["prediction_std"],
        "correlation": report["pairwise_prediction_correlation"],
    }), flush=True)


if __name__ == "__main__":
    main()
