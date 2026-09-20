"""Train a five temporal-fold x three-seed TabM385-k32 raw/EMA ensemble."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
import train_tabm385_k32_temporal3fold_3seed as base

recipe = base.recipe
RUN_NAME = "tabm385_k32_temporal5fold_3seed_ema0999"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "submissions" / RUN_NAME
OUTPUT_DIR = PROJECT_DIR / "outputs" / "submissions"
PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions" / RUN_NAME
METADATA_PATH = PROJECT_DIR / "outputs" / "submission_metadata" / f"{RUN_NAME}.json"
FOLD_ENDS = (42, 47, 52, 57, 62)
SEEDS = (42, 137, 2026)
EMA_DECAY = 0.999
VALID_MONTHS = (63, 64, 65, 67, 68, 69, 70)


def cosine_score(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(target @ prediction / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30))


def copy_parameters(model, values):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(values[name])


def train_one_model(fold_end, seed, train_features, test_features, months, target,
                    feature_columns, test_ids, valid_indices, device):
    fold_dir = RUN_DIR / f"train000_{fold_end:03d}"
    model_dir = fold_dir / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_test_path = PREDICTION_DIR / f"fold_end{fold_end}_seed{seed}_raw.feather"
    ema_test_path = PREDICTION_DIR / f"fold_end{fold_end}_seed{seed}_ema.feather"
    valid_path = model_dir / "validation_predictions.feather"
    result_path = model_dir / "result.json"
    if all(path.exists() for path in (raw_test_path, ema_test_path, valid_path, result_path)):
        raw_frame = pd.read_feather(raw_test_path).sort_values("sample_id")
        ema_frame = pd.read_feather(ema_test_path).sort_values("sample_id")
        valid_frame = pd.read_feather(valid_path).sort_values("sample_id")
        if not np.array_equal(raw_frame.sample_id.to_numpy(), test_ids):
            raise AssertionError("Saved raw test IDs do not align")
        if not np.array_equal(ema_frame.sample_id.to_numpy(), test_ids):
            raise AssertionError("Saved EMA test IDs do not align")
        return (raw_frame.prediction.to_numpy(np.float64),
                ema_frame.prediction.to_numpy(np.float64), valid_frame,
                json.loads(result_path.read_text(encoding="utf-8")))

    train_indices = np.flatnonzero(months <= fold_end)
    preprocessor = base.fit_or_load_preprocessor(
        fold_dir, train_features, train_indices, feature_columns, device
    )
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    scaled_target = ((target - target_mean) / target_std).astype(np.float32)
    recipe.seed_everything(seed)
    model = base.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.LEARNING_RATE, weight_decay=recipe.WEIGHT_DECAY
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ema_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    checkpoint_path = model_dir / "training_checkpoint.pt"
    logs = []
    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        ema_parameters = {name: value.to(device) for name, value in checkpoint["ema_parameters"].items()}
        logs = checkpoint["logs"]
        start_epoch = int(checkpoint["next_epoch"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        print(f"resume fold_end={fold_end} seed={seed} epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, recipe.EPOCHS + 1):
        started = time.perf_counter()
        learning_rate = recipe.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        shuffled = np.random.default_rng(seed + epoch).permutation(train_indices)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(shuffled), recipe.BATCH_SIZE):
            indices = shuffled[start:start + recipe.BATCH_SIZE]
            batch = preprocessor.transform(train_features[indices])
            batch_target = torch.as_tensor(scaled_target[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                members = model(batch).squeeze(-1)
                loss = F.mse_loss(members.float(), batch_target[:, None].expand_as(members))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    ema_parameters[name].lerp_(parameter.detach(), 1.0 - EMA_DECAY)
            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total_rows += count
        log = {"epoch": epoch, "learning_rate": learning_rate,
               "train_mse": total_loss / total_rows,
               "seconds": time.perf_counter() - started}
        logs.append(log)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "ema_parameters": {name: value.detach().cpu() for name, value in ema_parameters.items()},
            "next_epoch": epoch + 1, "logs": logs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }, temporary)
        temporary.replace(checkpoint_path)
        print(f"fold_end={fold_end} seed={seed} epoch={epoch:02d}/{recipe.EPOCHS} mse={log['train_mse']:.7f} seconds={log['seconds']:.1f}", flush=True)

    raw_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    all_test_indices = np.arange(len(test_features), dtype=np.int64)
    raw_test = recipe.predict(model, test_features, all_test_indices, preprocessor, amp_dtype).astype(np.float64) * target_std
    raw_valid = recipe.predict(model, train_features, valid_indices, preprocessor, amp_dtype).astype(np.float64) * target_std
    copy_parameters(model, ema_parameters)
    ema_test = recipe.predict(model, test_features, all_test_indices, preprocessor, amp_dtype).astype(np.float64) * target_std
    ema_valid = recipe.predict(model, train_features, valid_indices, preprocessor, amp_dtype).astype(np.float64) * target_std
    copy_parameters(model, raw_parameters)
    for values in (raw_test, raw_valid, ema_test, ema_valid):
        if not np.isfinite(values).all():
            raise AssertionError("Nonfinite prediction")
    pd.DataFrame({"sample_id": test_ids, "prediction": raw_test}).to_feather(raw_test_path)
    pd.DataFrame({"sample_id": test_ids, "prediction": ema_test}).to_feather(ema_test_path)
    valid_frame = pd.DataFrame({
        "sample_id": base.np.asarray(base.pd.read_feather(PROJECT_DIR / "data" / "raw" / "label.feather", columns=["sample_id"]).sort_values("sample_id").sample_id.to_numpy()[valid_indices]),
        "month": months[valid_indices], "target": target[valid_indices],
        "raw": raw_valid, "ema": ema_valid,
    })
    valid_frame.to_feather(valid_path)
    result = {
        "fold_end": fold_end, "seed": seed, "train_rows": int(len(train_indices)),
        "epochs": recipe.EPOCHS, "ema_decay": EMA_DECAY,
        "target_mean": target_mean, "target_std": target_std,
        "raw_test_std": float(raw_test.std()), "ema_test_std": float(ema_test.std()),
        "raw_valid_cosine": cosine_score(target[valid_indices], raw_valid),
        "ema_valid_cosine": cosine_score(target[valid_indices], ema_valid),
        "raw_test_path": str(raw_test_path), "ema_test_path": str(ema_test_path),
        "validation_path": str(valid_path), "logs": logs,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), model_dir / "raw_final_state.pt")
    torch.save({name: value.detach().cpu() for name, value in ema_parameters.items()}, model_dir / "ema_final_parameters.pt")
    del model, optimizer, scaler, preprocessor, raw_parameters, ema_parameters
    torch.cuda.empty_cache(); gc.collect()
    return raw_test, ema_test, valid_frame, result


def weighted_fold_average(fold_arrays, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return sum(weights[i] * fold_arrays[fold] for i, fold in enumerate(FOLD_ENDS))


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    for path in (RUN_DIR, OUTPUT_DIR, PREDICTION_DIR, METADATA_PATH.parent):
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)
    labels, train_features, test_features, template, feature_columns = base.load_features()
    months = labels.month.to_numpy(np.int16, copy=True)
    target = labels.target.to_numpy(np.float32, copy=True)
    test_ids = template.sample_id.to_numpy(copy=True)
    valid_indices = np.flatnonzero(np.isin(months, VALID_MONTHS))
    valid_target = target[valid_indices].astype(np.float64)
    valid_month = months[valid_indices]
    device = torch.device("cuda")

    raw_test_by_fold, ema_test_by_fold = {}, {}
    raw_valid_by_fold, ema_valid_by_fold = {}, {}
    model_results = []
    all_raw_test, all_ema_test = [], []
    for fold_end in FOLD_ENDS:
        fold_raw_test, fold_ema_test, fold_raw_valid, fold_ema_valid = [], [], [], []
        for seed in SEEDS:
            raw_test, ema_test, valid_frame, result = train_one_model(
                fold_end, seed, train_features, test_features, months, target,
                feature_columns, test_ids, valid_indices, device
            )
            fold_raw_test.append(raw_test); fold_ema_test.append(ema_test)
            fold_raw_valid.append(valid_frame.raw.to_numpy(np.float64))
            fold_ema_valid.append(valid_frame.ema.to_numpy(np.float64))
            all_raw_test.append(raw_test); all_ema_test.append(ema_test)
            model_results.append(result)
        raw_test_by_fold[fold_end] = np.mean(fold_raw_test, axis=0)
        ema_test_by_fold[fold_end] = np.mean(fold_ema_test, axis=0)
        raw_valid_by_fold[fold_end] = np.mean(fold_raw_valid, axis=0)
        ema_valid_by_fold[fold_end] = np.mean(fold_ema_valid, axis=0)
        for kind, values in (("raw", raw_test_by_fold[fold_end]), ("ema", ema_test_by_fold[fold_end])):
            pd.DataFrame({"sample_id": test_ids, "prediction": values}).to_csv(
                OUTPUT_DIR / f"{RUN_NAME}_fold_end{fold_end}_{kind}.csv", index=False
            )

    schemes = {
        "equal": [1, 1, 1, 1, 1],
        "linear_recent": [1, 2, 3, 4, 5],
        "exp_recent": [1, 1.5, 2.25, 3.375, 5.0625],
        "last3_equal": [0, 0, 1, 1, 1],
        "last3_recent": [0, 0, 1, 2, 3],
    }
    selection_mask = np.isin(valid_month, [63, 64, 65])
    forward_mask = np.isin(valid_month, [67, 68, 69, 70])
    candidate_rows = []
    candidate_test = {}
    for scheme_name, weights in schemes.items():
        raw_valid = weighted_fold_average(raw_valid_by_fold, weights)
        ema_valid = weighted_fold_average(ema_valid_by_fold, weights)
        raw_test = weighted_fold_average(raw_test_by_fold, weights)
        ema_test = weighted_fold_average(ema_test_by_fold, weights)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            valid_prediction = (1.0 - alpha) * raw_valid + alpha * ema_valid
            test_prediction = (1.0 - alpha) * raw_test + alpha * ema_test
            key = f"{scheme_name}_ema{int(alpha * 100):03d}"
            monthly = {str(month): cosine_score(valid_target[valid_month == month], valid_prediction[valid_month == month]) for month in VALID_MONTHS}
            row = {
                "key": key, "fold_weight_scheme": scheme_name, "fold_weights": weights,
                "ema_prediction_weight": alpha,
                "selection_63_65": cosine_score(valid_target[selection_mask], valid_prediction[selection_mask]),
                "forward_67_70": cosine_score(valid_target[forward_mask], valid_prediction[forward_mask]),
                "all_63_70_ex66": cosine_score(valid_target, valid_prediction),
                "monthly": monthly,
            }
            row["robust_score"] = 0.4 * row["selection_63_65"] + 0.6 * row["forward_67_70"]
            candidate_rows.append(row); candidate_test[key] = test_prediction
    selected = max(candidate_rows, key=lambda row: row["robust_score"])
    named_outputs = {
        "raw_equal": candidate_test["equal_ema000"],
        "ema_equal": candidate_test["equal_ema100"],
        "ema_recency": candidate_test["linear_recent_ema100"],
        "ema_last3": candidate_test["last3_equal_ema100"],
        "selected": candidate_test[selected["key"]],
    }
    output_paths = {}
    for name, prediction in named_outputs.items():
        path = OUTPUT_DIR / f"{RUN_NAME}_{name}.csv"
        pd.DataFrame({"sample_id": test_ids, "prediction": prediction}).to_csv(path, index=False)
        output_paths[name] = str(path)
    metadata = {
        "run_name": RUN_NAME, "model": "TabM385 k32", "fold_ends": list(FOLD_ENDS),
        "seeds": list(SEEDS), "model_count": len(FOLD_ENDS) * len(SEEDS),
        "epochs_per_model": recipe.EPOCHS, "ema_decay": EMA_DECAY,
        "validation_months": list(VALID_MONTHS), "selection_rule": "max 0.4*63-65 + 0.6*67-70",
        "selected": selected, "candidates": candidate_rows,
        "outputs": output_paths, "models": model_results,
        "raw_pairwise_test_correlation": np.corrcoef(np.column_stack(all_raw_test), rowvar=False).tolist(),
        "ema_pairwise_test_correlation": np.corrcoef(np.column_stack(all_ema_test), rowvar=False).tolist(),
        "submission_status": "prepared_not_submitted",
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "outputs": output_paths}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
