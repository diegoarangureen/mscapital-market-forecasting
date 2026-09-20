"""TabM385 with reversible per-sample RV60 target scaling."""

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
REFERENCE_DIR = PROJECT_DIR / "data" / "interim" / "kaggle_kernels" / "relative319_xs_tabm_notebook"
sys.path.insert(0, str(REFERENCE_DIR))

import run_extracted as recipe
import exp_gru_003_strong_joint as gru
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_COLUMNS

STATE_COLUMNS = "mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 realized_volatility_60 mid_momentum_60".split()
EXPERIMENT_ID = "EXP-TABM-038-RV60-TARGET-SCALE"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
FOLD_NAME = "train059_valid6270_ex66"
SCORE_KEY = "months_62_70_without_66"


def make_model_k32(input_dimension: int, device: torch.device):
    return recipe.TabM.make(
        n_num_features=input_dimension,
        cat_cardinalities=None,
        d_out=1,
        k=32,
        n_blocks=2,
        d_block=256,
        dropout=0.1,
        arch_type="tabm",
    ).to(device)


def run_scaled_model(features, target, months, rv60, device):
    train_indices = np.flatnonzero(months <= 59)
    valid_indices = np.flatnonzero((months >= 62) & (months <= 70))
    knots, medians, missing_columns = recipe.fit_quantile_knots(features, train_indices)
    preprocessor = recipe.QuantilePreprocessor(knots, medians, missing_columns, device)

    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    train_rv = rv60[train_indices]
    usable_train_rv = train_rv[np.isfinite(train_rv) & (train_rv > 0)]
    rv_median = float(np.median(usable_train_rv))
    clean_rv = np.where(np.isfinite(rv60) & (rv60 > 0), rv60, rv_median)
    sample_scale = np.clip(clean_rv / rv_median, 0.5, 2.0).astype(np.float32)
    scaled_target = ((target - target_mean) / (target_std * sample_scale)).astype(np.float32)

    recipe.seed_everything(recipe.SEED)
    model = make_model_k32(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.LEARNING_RATE, weight_decay=recipe.WEIGHT_DECAY
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    logs = []
    started = time.perf_counter()
    for epoch in range(1, recipe.EPOCHS + 1):
        rate = recipe.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        shuffled = np.random.default_rng(recipe.SEED + epoch).permutation(train_indices)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(shuffled), recipe.BATCH_SIZE):
            batch_indices = shuffled[start : start + recipe.BATCH_SIZE]
            batch = preprocessor.transform(features[batch_indices])
            batch_target = torch.as_tensor(
                scaled_target[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                member_predictions = model(batch).squeeze(-1)
                loss = F.mse_loss(
                    member_predictions.float(),
                    batch_target[:, None].expand_as(member_predictions),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(batch_indices)
            total_count += len(batch_indices)
        row = {
            "epoch": epoch,
            "learning_rate": rate,
            "train_mse": total_loss / total_count,
        }
        logs.append(row)
        print(f"rv60_target_scale epoch={epoch:02d} loss={row['train_mse']:.7f}", flush=True)

    normalized_prediction = recipe.predict(
        model, features, valid_indices, preprocessor, amp_dtype
    ).astype(np.float64)
    prediction = normalized_prediction * target_std * sample_scale[valid_indices]
    metrics = recipe.evaluate(
        target[valid_indices].astype(np.float64),
        prediction,
        months[valid_indices],
    )
    result = {
        "name": "tabm385_k32_rv60_target_scale",
        "feature_count": int(features.shape[1]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "seconds": time.perf_counter() - started,
        "target_mean": target_mean,
        "target_std": target_std,
        "rv60_train_median": rv_median,
        "scale_clip": [0.5, 2.0],
        "logs": logs,
        "metrics": metrics,
        "validation_indices": valid_indices,
        "prediction": prediction,
        "validation_scale": sample_scale[valid_indices],
    }
    del model, optimizer, preprocessor, knots, medians
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = torch.device("cuda")
    fold_dir = RUN_DIR / FOLD_NAME
    fold_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_feather(
        PROJECT_DIR / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    sample_ids = labels["sample_id"].to_numpy(copy=True)
    months = labels["month"].to_numpy(dtype=np.int16, copy=True)
    target = labels["target"].to_numpy(dtype=np.float32, copy=True)

    base, base_columns = gru.load_relative319(PROJECT_DIR, labels)
    xs40, xs_columns = recipe.add_relative_features(base, months, base_columns)
    state = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_market_microstructure_features.feather",
        columns=["sample_id", *STATE_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    order_frame = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_order_quote_position_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(sample_ids, state["sample_id"].to_numpy()):
        raise AssertionError("State sample IDs do not match labels.")
    if not np.array_equal(sample_ids, order_frame["sample_id"].to_numpy()):
        raise AssertionError("Order sample IDs do not match labels.")

    state_values = state[STATE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    rv60 = state["realized_volatility_60"].to_numpy(dtype=np.float32, copy=True)
    order_values = order_frame[ORDER_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    features = np.concatenate([base, xs40, order_values, state_values], axis=1)
    if features.shape[1] != 385:
        raise AssertionError(f"Expected 385 features, got {features.shape[1]}.")
    del state, order_frame, labels, base, xs40, order_values, state_values
    gc.collect()

    baseline_path = (
        PROJECT_DIR / "data" / "interim" / "tree_experiments"
        / "EXP-TABM-032-MARKET385-K32" / FOLD_NAME / "validation_predictions.feather"
    )
    baseline_frame = pd.read_feather(baseline_path).sort_values("sample_id").reset_index(drop=True)
    result = run_scaled_model(features, target, months, rv60, device)
    valid_indices = result["validation_indices"]
    if not np.array_equal(sample_ids[valid_indices], baseline_frame["sample_id"].to_numpy()):
        raise AssertionError("Baseline validation rows do not match.")

    baseline_prediction = baseline_frame["candidate"].to_numpy(dtype=np.float64, copy=True)
    baseline_score = float(recipe.evaluate(
        target[valid_indices].astype(np.float64),
        baseline_prediction,
        months[valid_indices],
    )[SCORE_KEY])
    candidate_score = float(result["metrics"][SCORE_KEY])
    delta = candidate_score - baseline_score

    pd.DataFrame({
        "sample_id": sample_ids[valid_indices],
        "month": months[valid_indices],
        "target": target[valid_indices],
        "baseline": baseline_prediction,
        "candidate": result["prediction"],
        "rv60_scale": result["validation_scale"],
    }).to_feather(fold_dir / "validation_predictions.feather")

    compact = {key: value for key, value in result.items()
               if key not in {"validation_indices", "prediction", "validation_scale"}}
    (fold_dir / "candidate_result.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "baseline": "TabM385 k32 standard global target scaling",
        "candidate": "TabM385 k32 with reversible per-sample RV60 target scaling",
        "fold": FOLD_NAME,
        "score_key": SCORE_KEY,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta": delta,
        "scale_formula": "clip(RV60 / train_median_RV60, 0.5, 2.0)",
        "pass_threshold": 0.001,
        "passed": bool(delta >= 0.001),
        "next_action": "paired_early_window" if delta >= 0.001 else "reject_rv60_target_scaling",
    }
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
