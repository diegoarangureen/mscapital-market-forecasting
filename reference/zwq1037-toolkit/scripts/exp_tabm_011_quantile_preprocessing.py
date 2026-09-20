"""TabM with training-fold empirical normal-quantile preprocessing.

Only preprocessing changes versus EXP-TABM-001. Architecture, optimizer,
fixed learning rate, batch size, seed, target scaling, and early stopping stay
unchanged. Two nested forward windows are evaluated independently.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as tabm
from audit_tabm_member_aggregation import predict_members


EXPERIMENT_ID = "EXP-TABM-011-QUANTILE"
N_QUANTILES = 1001
FIT_SAMPLE_ROWS = 200_000
PROBABILITY_EPSILON = 1.0e-4
FOLDS = {
    "train049_valid5059": {
        "inner_train_end": 39,
        "inner_valid_start": 40,
        "inner_valid_end": 49,
        "formal_train_end": 49,
        "formal_valid_start": 50,
        "formal_valid_end": 59,
    },
    "train059_valid6070": {
        "inner_train_end": 49,
        "inner_valid_start": 50,
        "inner_valid_end": 59,
        "formal_train_end": 59,
        "formal_valid_start": 60,
        "formal_valid_end": 70,
    },
}


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def evaluate(
    target: np.ndarray,
    prediction: np.ndarray,
    months: np.ndarray,
    valid_start: int,
    valid_end: int,
) -> dict[str, object]:
    month_values = np.arange(valid_start, valid_end + 1)
    monthly = [
        {
            "month": int(month),
            "rows": int((months == month).sum()),
            "cosine": cosine(target[months == month], prediction[months == month]),
        }
        for month in month_values
    ]
    monthly_values = np.asarray([row["cosine"] for row in monthly])
    midpoint = (valid_start + valid_end) // 2
    result: dict[str, object] = {
        "overall": cosine(target, prediction),
        "first_half": cosine(target[months <= midpoint], prediction[months <= midpoint]),
        "second_half": cosine(target[months > midpoint], prediction[months > midpoint]),
        "first_half_months": f"{valid_start}-{midpoint}",
        "second_half_months": f"{midpoint + 1}-{valid_end}",
        "monthly_macro_mean": float(monthly_values.mean()),
        "monthly_std": float(monthly_values.std(ddof=0)),
        "monthly_worst": float(monthly_values.min()),
        "monthly_q25": float(np.quantile(monthly_values, 0.25)),
        "monthly": monthly,
    }
    if valid_start <= 66 <= valid_end:
        no66 = months != 66
        result["without_month_66"] = cosine(target[no66], prediction[no66])
    if valid_end >= 70:
        recent = months >= 67
        result["months_67_70"] = cosine(target[recent], prediction[recent])
    return result


def fit_quantile_knots(
    features: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    count = min(FIT_SAMPLE_ROWS, len(train_indices))
    sampled_indices = generator.choice(train_indices, size=count, replace=False)
    probabilities = np.linspace(0.0, 1.0, N_QUANTILES, dtype=np.float64)
    feature_count = features.shape[1]
    knots = np.empty((feature_count, N_QUANTILES), dtype=np.float32)
    medians = np.empty(feature_count, dtype=np.float32)
    has_missing = np.zeros(feature_count, dtype=bool)
    for column_index in range(feature_count):
        full_values = features[train_indices, column_index]
        finite_full = full_values[np.isfinite(full_values)]
        has_missing[column_index] = len(finite_full) != len(full_values)
        if len(finite_full) == 0:
            median = 0.0
        else:
            median = float(np.median(finite_full))
        medians[column_index] = median
        sampled_values = features[sampled_indices, column_index]
        sampled_values = sampled_values[np.isfinite(sampled_values)]
        if len(sampled_values) == 0:
            knots[column_index] = median
        else:
            knots[column_index] = np.quantile(sampled_values, probabilities).astype(
                np.float32
            )
    missing_columns = np.flatnonzero(has_missing).astype(np.int64)
    return knots, medians, missing_columns


class QuantileBatchPreprocessor:
    """Apply an empirical CDF and inverse-normal map on each GPU batch."""

    def __init__(
        self,
        knots: np.ndarray,
        medians: np.ndarray,
        missing_columns: np.ndarray,
        device: torch.device,
    ) -> None:
        self.knots = torch.as_tensor(knots, dtype=torch.float32, device=device)
        self.medians = torch.as_tensor(medians, dtype=torch.float32, device=device)
        self.missing_columns = torch.as_tensor(
            missing_columns, dtype=torch.int64, device=device
        )
        self.device = device

    @property
    def output_dimension(self) -> int:
        return int(self.medians.numel() + self.missing_columns.numel())

    def transform(self, raw_batch: np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(raw_batch, dtype=torch.float32, device=self.device)
        missing_mask = ~torch.isfinite(values)
        values = torch.where(missing_mask, self.medians, values)
        transposed = values.transpose(0, 1).contiguous()

        left = torch.searchsorted(self.knots, transposed, right=False)
        right = torch.searchsorted(self.knots, transposed, right=True)
        maximum_index = self.knots.shape[1] - 1
        exact_index = left.clamp(0, maximum_index)
        exact_value = torch.gather(self.knots, 1, exact_index)
        is_exact = (left <= maximum_index) & (exact_value == transposed)

        upper_index = left.clamp(1, maximum_index)
        lower_index = upper_index - 1
        lower_value = torch.gather(self.knots, 1, lower_index)
        upper_value = torch.gather(self.knots, 1, upper_index)
        fraction = (transposed - lower_value) / (upper_value - lower_value).clamp_min(
            1.0e-12
        )
        interpolated_position = lower_index.float() + fraction.clamp(0.0, 1.0)
        exact_position = 0.5 * (left.float() + right.float() - 1.0)
        position = torch.where(is_exact, exact_position, interpolated_position)
        probability = (position / maximum_index).clamp(
            PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        normal = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
        output = normal.transpose(0, 1).contiguous()
        if self.missing_columns.numel() > 0:
            indicators = missing_mask.index_select(1, self.missing_columns).to(
                output.dtype
            )
            output = torch.cat([output, indicators], dim=1)
        return output


def scaled_targets(
    target: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, float]:
    mean = float(target[train_indices].mean(dtype=np.float64))
    standard_deviation = float(target[train_indices].std(dtype=np.float64))
    scaled = ((target - mean) / standard_deviation).astype(np.float32)
    return scaled, standard_deviation


def select_epoch(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    config: dict[str, int],
    device: torch.device,
    fold_dir: Path,
) -> tuple[int, list[dict[str, float]], float]:
    selection_path = fold_dir / "epoch_selection.json"
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        return int(saved["best_epoch"]), saved["logs"], float(saved["seconds"])
    train_indices = np.flatnonzero(months <= config["inner_train_end"])
    valid_indices = np.flatnonzero(
        (months >= config["inner_valid_start"])
        & (months <= config["inner_valid_end"])
    )
    knots, medians, missing_columns = fit_quantile_knots(
        features, train_indices, tabm.SEED
    )
    preprocessor = QuantileBatchPreprocessor(
        knots, medians, missing_columns, device
    )
    target_scaled, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    best_epoch = 1
    best_score = -np.inf
    stale_epochs = 0
    logs: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, tabm.MAX_SCOUT_EPOCHS + 1):
        loss = tabm.train_one_epoch(
            model,
            optimizer,
            features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        prediction = tabm.predict(model, features, valid_indices, preprocessor)
        prediction *= target_std
        score = cosine(target[valid_indices], prediction)
        if score > best_score + tabm.MIN_DELTA:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        logs.append(
            {
                "epoch": epoch,
                "train_mse": float(loss),
                "validation_cosine": score,
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
        )
        print(
            f"quantile inner epoch={epoch:02d} cosine={score:.8f} "
            f"best={best_score:.8f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= tabm.PATIENCE:
            break
    seconds = time.perf_counter() - started
    selection_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "seconds": seconds,
                "logs": logs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, optimizer, preprocessor, knots, medians
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, logs, seconds


def train_formal(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    config: dict[str, int],
    best_epoch: int,
    device: torch.device,
    fold_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    prediction_path = fold_dir / "quantile_predictions.feather"
    details_path = fold_dir / "formal_details.json"
    if prediction_path.exists() and details_path.exists():
        frame = pd.read_feather(prediction_path)
        return (
            frame["mean_prediction"].to_numpy(dtype=np.float64),
            frame["trim1_prediction"].to_numpy(dtype=np.float64),
            json.loads(details_path.read_text(encoding="utf-8")),
        )
    train_indices = np.flatnonzero(months <= config["formal_train_end"])
    valid_indices = np.flatnonzero(
        (months >= config["formal_valid_start"])
        & (months <= config["formal_valid_end"])
    )
    knots, medians, missing_columns = fit_quantile_knots(
        features, train_indices, tabm.SEED
    )
    np.savez_compressed(
        fold_dir / "quantile_preprocessing.npz",
        knots=knots,
        medians=medians,
        missing_columns=missing_columns,
    )
    preprocessor = QuantileBatchPreprocessor(
        knots, medians, missing_columns, device
    )
    target_scaled, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    logs = []
    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        loss = tabm.train_one_epoch(
            model,
            optimizer,
            features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        logs.append({"epoch": epoch, "train_mse": float(loss)})
        print(
            f"quantile formal epoch={epoch:02d}/{best_epoch:02d} loss={loss:.8f}",
            flush=True,
        )
    seconds = time.perf_counter() - started
    members = predict_members(model, features[valid_indices], preprocessor, target_std)
    mean_prediction = members.mean(axis=1).astype(np.float64)
    trim1_prediction = np.sort(members, axis=1)[:, 1:-1].mean(axis=1).astype(
        np.float64
    )
    pd.DataFrame(
        {
            "mean_prediction": mean_prediction,
            "trim1_prediction": trim1_prediction,
        }
    ).to_feather(prediction_path)
    details = {
        "selected_epoch": best_epoch,
        "training_seconds": seconds,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(valid_indices)),
        "input_dimension": preprocessor.output_dimension,
        "missing_indicator_count": int(len(missing_columns)),
        "logs": logs,
    }
    details_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, optimizer, preprocessor, knots, medians, members
    torch.cuda.empty_cache()
    gc.collect()
    return mean_prediction, trim1_prediction, details


def load_baseline(
    project_dir: Path, fold_name: str, validation_rows: pd.DataFrame
) -> np.ndarray:
    if fold_name == "train049_valid5059":
        frame = pd.read_feather(
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
            / "validation_predictions.feather"
        )
        prediction_column = "tabm_mean"
    else:
        frame = pd.read_feather(
            project_dir / "outputs" / "predictions" / "exp-tabm-001_valid.feather"
        )
        prediction_column = "prediction"
    if not np.array_equal(
        frame["sample_id"].to_numpy(), validation_rows["sample_id"].to_numpy()
    ):
        raise AssertionError(f"Baseline rows are not aligned for {fold_name}.")
    return frame[prediction_column].to_numpy(dtype=np.float64)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    tabm.SEED = 42

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, feature_columns, _, _ = tabm.load_exp053r_data(project_dir)
    features = model_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    del model_data
    gc.collect()

    results: dict[str, object] = {}
    for fold_name, config in FOLDS.items():
        print(f"starting {fold_name}", flush=True)
        fold_dir = run_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_epoch, inner_logs, inner_seconds = select_epoch(
            features, target, months, config, torch.device("cuda"), fold_dir
        )
        mean_prediction, trim1_prediction, formal_details = train_formal(
            features,
            target,
            months,
            config,
            best_epoch,
            torch.device("cuda"),
            fold_dir,
        )
        valid_mask = (
            (months >= config["formal_valid_start"])
            & (months <= config["formal_valid_end"])
        )
        validation_rows = pd.DataFrame(
            {
                "sample_id": sample_ids[valid_mask],
                "month": months[valid_mask],
                "target": target[valid_mask],
            }
        )
        baseline = load_baseline(project_dir, fold_name, validation_rows)
        predictions = {
            "baseline_standardized_tabm": baseline,
            "quantile_tabm_mean": mean_prediction,
            "quantile_tabm_trim1": trim1_prediction,
        }
        metrics = {
            name: evaluate(
                validation_rows["target"].to_numpy(dtype=np.float64),
                prediction,
                validation_rows["month"].to_numpy(),
                config["formal_valid_start"],
                config["formal_valid_end"],
            )
            for name, prediction in predictions.items()
        }
        output = validation_rows.copy()
        for name, prediction in predictions.items():
            output[name] = prediction
        output.to_feather(fold_dir / "comparison_predictions.feather")
        summary = []
        baseline_overall = metrics["baseline_standardized_tabm"]["overall"]
        for name, values in metrics.items():
            summary.append(
                {
                    "candidate": name,
                    **{key: value for key, value in values.items() if key != "monthly"},
                    "overall_change_vs_baseline": values["overall"] - baseline_overall,
                }
            )
        pd.DataFrame(summary).to_csv(fold_dir / "summary.csv", index=False)
        monthly = pd.DataFrame(
            {
                "month": list(
                    range(
                        config["formal_valid_start"],
                        config["formal_valid_end"] + 1,
                    )
                ),
                **{
                    name: [row["cosine"] for row in values["monthly"]]
                    for name, values in metrics.items()
                },
            }
        )
        monthly.to_csv(fold_dir / "monthly_cosine.csv", index=False)
        results[fold_name] = {
            "protocol": config,
            "selected_epoch": best_epoch,
            "inner_seconds": inner_seconds,
            "inner_logs": inner_logs,
            "formal": formal_details,
            "metrics": metrics,
        }
        print(pd.DataFrame(summary).to_string(index=False), flush=True)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "single_variable_change": "standard scaling -> empirical normal quantile scaling",
        "parameters_held_fixed": {
            "learning_rate": tabm.LEARNING_RATE,
            "learning_rate_schedule": "constant",
            "weight_decay": tabm.WEIGHT_DECAY,
            "batch_size": tabm.BATCH_SIZE,
            "k": tabm.K,
            "n_blocks": tabm.N_BLOCKS,
            "d_block": tabm.D_BLOCK,
            "dropout": tabm.DROPOUT,
            "seed": tabm.SEED,
        },
        "quantile_parameters": {
            "n_quantiles": N_QUANTILES,
            "fit_sample_rows": FIT_SAMPLE_ROWS,
            "output_distribution": "normal",
            "probability_epsilon": PROBABILITY_EPSILON,
            "missing_handling": "training median plus existing binary indicators",
            "fit_scope": "each training fold only",
        },
        "folds": results,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
