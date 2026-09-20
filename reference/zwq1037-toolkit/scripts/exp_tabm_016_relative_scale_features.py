"""Test a predeclared 12-column relative-scale feature add-on for Quantile TabM.

The model, seed, 15-epoch budget, cosine learning-rate schedule, target scaling,
and XGBoost blend stay fixed. Only the input feature set changes from 307 to 319.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as tabm
from audit_tabm_member_aggregation import predict_members
from exp_tabm_011_quantile_preprocessing import (
    QuantileBatchPreprocessor,
    evaluate,
    fit_quantile_knots,
)
from exp_tabm_014_quantile_stage_curves import (
    FOLDS,
    load_xgboost_prediction,
    unit,
)
from exp_tabm_015_quantile_cosine_lr_curves import learning_rate_at


EXPERIMENT_ID = "EXP-TABM-016-RELATIVE-SCALE-ADD12"
MAX_EPOCHS = 15
SEED = 42
PRICE_RELATIVE_COLUMNS = [
    "mid_last_vs_full_mean",
    "mid_last_vs_60_mean",
    "relative_spread_mean",
    "relative_spread_std",
    "relative_spread_last",
    "relative_spread_60_mean",
    "trade_price_vs_mid_mean",
    "trade_price_vs_mid_last",
]
DERIVED_RELATIVE_COLUMNS = [
    "x_recent60_trade_count_share",
    "x_recent60_trade_volume_share",
    "x_trade_count_peak_share",
    "x_trade_volume_peak_share",
]
RELATIVE_COLUMNS = PRICE_RELATIVE_COLUMNS + DERIVED_RELATIVE_COLUMNS


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Return a finite float32 ratio, with invalid denominators left missing."""

    numerator_values = numerator.to_numpy(dtype=np.float64, copy=False)
    denominator_values = denominator.to_numpy(dtype=np.float64, copy=False)
    valid = (
        np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
        & (np.abs(denominator_values) > 1.0e-12)
    )
    output = np.full(len(numerator_values), np.nan, dtype=np.float32)
    np.divide(
        numerator_values,
        denominator_values,
        out=output,
        where=valid,
        casting="unsafe",
    )
    return output


def add_relative_features(project_dir: Path, model_data: pd.DataFrame) -> None:
    """Attach only the predeclared target-free train features in sample-ID order."""

    price = pd.read_feather(
        project_dir / "data" / "processed" / "train_market_microstructure_features.feather",
        columns=["sample_id", *PRICE_RELATIVE_COLUMNS],
    ).sort_values("sample_id")
    if not np.array_equal(
        model_data["sample_id"].to_numpy(), price["sample_id"].to_numpy()
    ):
        raise AssertionError("Relative-price feature IDs do not match EXP053R rows.")
    for column in PRICE_RELATIVE_COLUMNS:
        model_data[column] = price[column].to_numpy(dtype=np.float32, copy=False)
    del price

    model_data["x_recent60_trade_count_share"] = safe_ratio(
        model_data["last60_transaction_count_sum"],
        model_data["transaction_count_sum"],
    )
    model_data["x_recent60_trade_volume_share"] = safe_ratio(
        model_data["last60_transaction_volume_sum"],
        model_data["transaction_volume_sum"],
    )
    model_data["x_trade_count_peak_share"] = safe_ratio(
        model_data["transaction_count_max"],
        model_data["transaction_count_sum"],
    )
    model_data["x_trade_volume_peak_share"] = safe_ratio(
        model_data["transaction_volume_max"],
        model_data["transaction_volume_sum"],
    )


def load_baseline_row(project_dir: Path, fold_name: str) -> dict[str, float]:
    curve = pd.read_csv(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-015-QUANTILE-COSINE-LR-CURVES"
        / fold_name
        / "epoch_curve.csv"
    )
    row = curve.loc[curve["epoch"] == MAX_EPOCHS]
    if len(row) != 1:
        raise AssertionError(f"Expected one epoch-{MAX_EPOCHS} baseline row.")
    return {key: float(value) for key, value in row.iloc[0].items()}


def restore_rng(checkpoint: dict) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])


def run_fold(
    project_dir: Path,
    fold_name: str,
    train_end: int,
    valid_start: int,
    valid_end: int,
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    sample_ids: np.ndarray,
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "result.json"
    prediction_path = fold_dir / "validation_predictions.feather"
    if result_path.exists() and prediction_path.exists():
        print(f"reusing completed {fold_name}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    train_indices = np.flatnonzero(months <= train_end)
    valid_indices = np.flatnonzero((months >= valid_start) & (months <= valid_end))
    validation_target = target[valid_indices].astype(np.float64)
    validation_months = months[valid_indices]
    validation_ids = sample_ids[valid_indices]

    preprocessing_path = fold_dir / "quantile_preprocessing.npz"
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
    else:
        knots, medians, missing_columns = fit_quantile_knots(
            features, train_indices, tabm.SEED
        )
        np.savez_compressed(
            preprocessing_path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
        )
    preprocessor = QuantileBatchPreprocessor(knots, medians, missing_columns, device)
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_std).astype(np.float32)

    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tabm.LEARNING_RATE,
        weight_decay=tabm.WEIGHT_DECAY,
    )
    checkpoint_path = fold_dir / "checkpoint.pt"
    logs: list[dict[str, float]] = []
    starting_epoch = 1
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logs = checkpoint["logs"]
        starting_epoch = int(checkpoint["epoch"]) + 1
        restore_rng(checkpoint)
        print(f"resuming {fold_name} at epoch {starting_epoch}", flush=True)

    for epoch in range(starting_epoch, MAX_EPOCHS + 1):
        learning_rate = learning_rate_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        loss = tabm.train_one_epoch(
            model,
            optimizer,
            features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        logs.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_mse": float(loss),
            }
        )
        tabm.atomic_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "logs": logs,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_path,
        )
        print(
            f"{fold_name} epoch={epoch:02d}/{MAX_EPOCHS} "
            f"lr={learning_rate:.7f} loss={loss:.8f}",
            flush=True,
        )

    members = predict_members(
        model, features[valid_indices], preprocessor, target_std
    ).astype(np.float64)
    mean_prediction = members.mean(axis=1)
    trim1_prediction = np.sort(members, axis=1)[:, 1:-1].mean(axis=1)
    xgboost = load_xgboost_prediction(project_dir, fold_name, validation_ids)
    predictions = {
        "tabm_mean": mean_prediction,
        "tabm_trim1": trim1_prediction,
        "b001_mean": 0.75 * unit(mean_prediction) + 0.25 * unit(xgboost),
        "b001_trim1": 0.75 * unit(trim1_prediction) + 0.25 * unit(xgboost),
    }
    metrics = {
        name: evaluate(
            validation_target,
            prediction,
            validation_months,
            valid_start,
            valid_end,
        )
        for name, prediction in predictions.items()
    }
    baseline = load_baseline_row(project_dir, fold_name)
    metrics["tabm_mean"]["overall_delta_vs_307"] = (
        metrics["tabm_mean"]["overall"] - baseline["tabm_overall"]
    )
    metrics["b001_mean"]["overall_delta_vs_307"] = (
        metrics["b001_mean"]["overall"] - baseline["b001_overall"]
    )

    output = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "month": validation_months,
            "target": validation_target,
            "xgboost": xgboost,
            **predictions,
        }
    )
    output.to_feather(prediction_path)
    monthly = pd.DataFrame(
        {
            "month": list(range(valid_start, valid_end + 1)),
            **{
                name: [item["cosine"] for item in values["monthly"]]
                for name, values in metrics.items()
            },
        }
    )
    monthly.to_csv(fold_dir / "monthly_cosine.csv", index=False)

    result = {
        "fold": fold_name,
        "train_months": f"0-{train_end}",
        "validation_months": f"{valid_start}-{valid_end}",
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(valid_indices)),
        "raw_feature_count": int(features.shape[1]),
        "input_dimension_after_missing_indicators": preprocessor.output_dimension,
        "epochs": MAX_EPOCHS,
        "logs": logs,
        "baseline_epoch15": baseline,
        "metrics": metrics,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_rows = []
    for name, values in metrics.items():
        summary_rows.append(
            {
                "candidate": name,
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"monthly", "first_half_months", "second_half_months"}
                },
            }
        )
    pd.DataFrame(summary_rows).to_csv(fold_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)

    del model, optimizer, preprocessor, members, knots, medians
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    tabm.SEED = SEED
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, feature_columns, _, _ = tabm.load_exp053r_data(project_dir)
    if any(column in feature_columns for column in RELATIVE_COLUMNS):
        raise AssertionError("Relative add-on unexpectedly overlaps the 307-column baseline.")
    add_relative_features(project_dir, model_data)
    candidate_columns = [*feature_columns, *RELATIVE_COLUMNS]
    if len(candidate_columns) != 319 or len(set(candidate_columns)) != 319:
        raise AssertionError("Expected exactly 319 unique model columns.")
    features = model_data[candidate_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    del model_data
    gc.collect()

    results = {}
    for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
        results[fold_name] = run_fold(
            project_dir,
            fold_name,
            train_end,
            valid_start,
            valid_end,
            features,
            target,
            months,
            sample_ids,
            run_dir,
            torch.device("cuda"),
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "single_variable_change": "add 12 predeclared target-free relative-scale features",
        "baseline": "EXP-TABM-015 epoch 15, same smooth cosine LR schedule",
        "baseline_feature_count": len(feature_columns),
        "candidate_feature_count": len(candidate_columns),
        "added_features": RELATIVE_COLUMNS,
        "fixed_parameters": {
            "seed": tabm.SEED,
            "epochs": MAX_EPOCHS,
            "initial_learning_rate": 2.0e-3,
            "minimum_learning_rate": 1.0e-4,
            "schedule_length_epochs": 20,
            "weight_decay": tabm.WEIGHT_DECAY,
            "batch_size": tabm.BATCH_SIZE,
            "k": tabm.K,
            "n_blocks": tabm.N_BLOCKS,
            "d_block": tabm.D_BLOCK,
            "dropout": tabm.DROPOUT,
        },
        "folds": results,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

