"""Leakage-safe forward backtest of the six-source Maximin blend.

The protocol is nested for every learned choice:
- inner train 0-39 / inner validation 40-49 selects TabM epochs;
- formal train 0-49 / formal validation 50-59 measures transfer;
- correlation pruning is recomputed from the corresponding training months only.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import exp_tabm_001_exp053r_features as tabm
from audit_tabm_member_aggregation import predict_members
from backtest_b001_train049_valid5059 import cosine, evaluate
from exp_tabm_003_cosine_loss import (
    INITIAL_LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY as COSINE_WEIGHT_DECAY,
    train_one_epoch_cosine,
)
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters


EXPERIMENT_ID = "EXP-BACKTEST-002-MAXIMIN-TRAIN049-VALID5059"
INNER_TRAIN_END = 39
INNER_VALID_START = 40
INNER_VALID_END = 49
FORMAL_TRAIN_END = 49
FORMAL_VALID_START = 50
FORMAL_VALID_END = 59
CORRELATION_THRESHOLD = 0.995
SAMPLES_PER_MONTH = 1000


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise AssertionError("Cannot normalize a zero vector.")
    return values / norm


def centered_unit(values: np.ndarray) -> np.ndarray:
    return unit(values - float(values.mean()))


def fit_preprocessing(
    features: np.ndarray, indices: np.ndarray, device: torch.device
) -> tuple[tabm.BatchPreprocessor, tuple[np.ndarray, ...]]:
    arrays = tabm.fit_preprocessor(features, indices)
    return tabm.BatchPreprocessor(*arrays, device), arrays


def scaled_targets(
    target: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, float, float]:
    mean = float(target[indices].mean(dtype=np.float64))
    standard_deviation = float(target[indices].std(dtype=np.float64))
    scaled = ((target - mean) / standard_deviation).astype(np.float32)
    return scaled, mean, standard_deviation


def train_xgb_for_importance(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    train_end: int,
) -> XGBRegressor:
    train_indices = np.flatnonzero(months <= train_end)
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    parameters["device"] = "cuda"
    parameters["n_jobs"] = 2
    model = XGBRegressor(**parameters)
    centered_target = target[train_indices] - float(
        target[train_indices].mean(dtype=np.float64)
    )
    print(f"training XGBoost importance model on months 0-{train_end}", flush=True)
    model.fit(features[train_indices], centered_target)
    return model


def select_correlated_features(
    features: np.ndarray,
    feature_columns: list[str],
    months: np.ndarray,
    train_end: int,
    importance_model: XGBRegressor,
) -> tuple[list[str], dict[str, object]]:
    random_generator = np.random.default_rng(42)
    sampled_indices = []
    for month in range(train_end + 1):
        month_indices = np.flatnonzero(months == month)
        sample_size = min(SAMPLES_PER_MONTH, len(month_indices))
        sampled_indices.append(
            random_generator.choice(month_indices, size=sample_size, replace=False)
        )
    selected_indices = np.concatenate(sampled_indices)
    sample = pd.DataFrame(
        features[selected_indices], columns=feature_columns, copy=False
    )
    correlation = sample.corr(method="pearson", min_periods=200).abs()
    del sample
    gc.collect()

    total_gain = importance_model.get_booster().get_score(importance_type="total_gain")
    ranked = sorted(
        feature_columns,
        key=lambda feature: (-total_gain.get(feature, 0.0), feature),
    )
    kept: list[str] = []
    removed: list[dict[str, object]] = []
    for feature in ranked:
        if not kept:
            kept.append(feature)
            continue
        correlations_to_kept = correlation.loc[feature, kept]
        highest = float(correlations_to_kept.max(skipna=True))
        if highest >= CORRELATION_THRESHOLD:
            keeper = str(correlations_to_kept.idxmax(skipna=True))
            removed.append(
                {
                    "removed_feature": feature,
                    "kept_feature": keeper,
                    "absolute_correlation": highest,
                    "removed_total_gain": float(total_gain.get(feature, 0.0)),
                    "kept_total_gain": float(total_gain.get(keeper, 0.0)),
                }
            )
        else:
            kept.append(feature)
    kept_set = set(kept)
    kept_in_source_order = [name for name in feature_columns if name in kept_set]
    details = {
        "selection_source": f"months 0-{train_end} only",
        "samples_per_month": SAMPLES_PER_MONTH,
        "sample_count": int(len(selected_indices)),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "original_feature_count": len(feature_columns),
        "kept_feature_count": len(kept_in_source_order),
        "removed_feature_count": len(removed),
        "kept_features_in_source_order": kept_in_source_order,
        "removed_pairs": removed,
    }
    print(
        f"corr selection 0-{train_end}: kept={len(kept_in_source_order)}, "
        f"removed={len(removed)}",
        flush=True,
    )
    return kept_in_source_order, details


def train_tabm_stage(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    train_end: int,
    valid_start: int,
    valid_end: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    fixed_epochs: int | None,
    device: torch.device,
) -> tuple[int, list[dict[str, float]], np.ndarray | None, np.ndarray | None, float]:
    train_indices = np.flatnonzero(months <= train_end)
    valid_indices = np.flatnonzero((months >= valid_start) & (months <= valid_end))
    preprocessor, _ = fit_preprocessing(features, train_indices, device)
    target_scaled, _, target_std = scaled_targets(target, train_indices)
    tabm.seed_everything(42)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_epoch = 1
    best_score = -np.inf
    stale_epochs = 0
    logs: list[dict[str, float]] = []
    epoch_limit = fixed_epochs if fixed_epochs is not None else max_epochs
    started = time.perf_counter()
    for epoch in range(1, epoch_limit + 1):
        if loss_name == "cosine":
            loss = train_one_epoch_cosine(
                model,
                optimizer,
                features,
                target_scaled,
                train_indices,
                preprocessor,
                epoch,
            )
        else:
            loss = tabm.train_one_epoch(
                model,
                optimizer,
                features,
                target_scaled,
                train_indices,
                preprocessor,
                epoch,
            )
        if fixed_epochs is None:
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
                    "train_loss": float(loss),
                    "validation_cosine": score,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                }
            )
            print(
                f"{loss_name} inner epoch={epoch:02d} cosine={score:.8f} "
                f"best={best_score:.8f}@{best_epoch}",
                flush=True,
            )
            if stale_epochs >= patience:
                break
        else:
            logs.append({"epoch": epoch, "train_loss": float(loss)})
            print(
                f"{loss_name} formal epoch={epoch:02d}/{epoch_limit:02d} "
                f"loss={loss:.8f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started

    mean_prediction = None
    trim1_prediction = None
    if fixed_epochs is not None:
        members = predict_members(
            model, features[valid_indices], preprocessor, target_std
        )
        mean_prediction = members.mean(axis=1)
        trim1_prediction = np.sort(members, axis=1)[:, 1:-1].mean(axis=1)
        best_epoch = fixed_epochs
    del model, optimizer, preprocessor
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, logs, mean_prediction, trim1_prediction, elapsed


def run_tabm_variant(
    name: str,
    all_features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    inner_column_indices: np.ndarray,
    formal_column_indices: np.ndarray,
    loss_name: str,
    device: torch.device,
    run_dir: Path,
) -> dict[str, object]:
    output_path = run_dir / f"{name}_predictions.feather"
    details_path = run_dir / f"{name}_details.json"
    if output_path.exists() and details_path.exists():
        print(f"reusing {name} predictions", flush=True)
        return json.loads(details_path.read_text(encoding="utf-8"))

    is_cosine = loss_name == "cosine"
    learning_rate = INITIAL_LEARNING_RATE if is_cosine else tabm.LEARNING_RATE
    weight_decay = COSINE_WEIGHT_DECAY if is_cosine else tabm.WEIGHT_DECAY
    max_epochs = MAX_EPOCHS if is_cosine else tabm.MAX_SCOUT_EPOCHS
    patience = 5 if is_cosine else tabm.PATIENCE
    best_epoch, inner_logs, _, _, inner_seconds = train_tabm_stage(
        all_features[:, inner_column_indices],
        target,
        months,
        INNER_TRAIN_END,
        INNER_VALID_START,
        INNER_VALID_END,
        max_epochs,
        patience,
        learning_rate,
        weight_decay,
        loss_name,
        None,
        device,
    )
    _, formal_logs, mean_prediction, trim1_prediction, formal_seconds = (
        train_tabm_stage(
            all_features[:, formal_column_indices],
            target,
            months,
            FORMAL_TRAIN_END,
            FORMAL_VALID_START,
            FORMAL_VALID_END,
            max_epochs,
            patience,
            learning_rate,
            weight_decay,
            loss_name,
            best_epoch,
            device,
        )
    )
    pd.DataFrame(
        {"mean_prediction": mean_prediction, "trim1_prediction": trim1_prediction}
    ).to_feather(output_path)
    details = {
        "name": name,
        "loss": loss_name,
        "inner_feature_count": int(len(inner_column_indices)),
        "formal_feature_count": int(len(formal_column_indices)),
        "selected_epoch": int(best_epoch),
        "inner_seconds": inner_seconds,
        "formal_seconds": formal_seconds,
        "inner_logs": inner_logs,
        "formal_logs": formal_logs,
    }
    details_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return details


def run_lightgbm(
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    feature_columns: list[str],
    run_dir: Path,
) -> dict[str, object]:
    output_path = run_dir / "lightgbm_predictions.feather"
    details_path = run_dir / "lightgbm_details.json"
    if output_path.exists() and details_path.exists():
        print("reusing LightGBM predictions", flush=True)
        return json.loads(details_path.read_text(encoding="utf-8"))
    train_indices = np.flatnonzero(months <= FORMAL_TRAIN_END)
    valid_indices = np.flatnonzero(
        (months >= FORMAL_VALID_START) & (months <= FORMAL_VALID_END)
    )
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    parameters = {
        "objective": "regression",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": 2,
        "verbosity": -1,
        "force_col_wise": True,
        "deterministic": True,
    }
    model = LGBMRegressor(**parameters)
    print("training LightGBM on months 0-49 with two CPU threads", flush=True)
    started = time.perf_counter()
    model.fit(features[train_indices], target[train_indices] - target_mean)
    elapsed = time.perf_counter() - started
    prediction = np.asarray(model.predict(features[valid_indices]), dtype=np.float64)
    pd.DataFrame({"prediction": prediction}).to_feather(output_path)
    joblib.dump(model, run_dir / "lightgbm_model.joblib")
    details = {
        "training_seconds": elapsed,
        "target_mean": target_mean,
        "feature_count": len(feature_columns),
        "parameters": parameters,
    }
    details_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    gc.collect()
    return details


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, feature_columns, _, _ = tabm.load_exp053r_data(project_dir)
    features = model_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    validation_mask = (months >= FORMAL_VALID_START) & (months <= FORMAL_VALID_END)
    validation_rows = model_data.loc[
        validation_mask, ["sample_id", "month", "target"]
    ].reset_index(drop=True)
    del model_data
    gc.collect()
    column_to_index = {name: index for index, name in enumerate(feature_columns)}

    selection_paths = {
        39: run_dir / "corr_selection_train039.json",
        49: run_dir / "corr_selection_train049.json",
    }
    selections: dict[int, dict[str, object]] = {}
    for train_end in (39, 49):
        selection_path = selection_paths[train_end]
        if selection_path.exists():
            selections[train_end] = json.loads(
                selection_path.read_text(encoding="utf-8")
            )
            continue
        if train_end == 49:
            importance_model = XGBRegressor()
            importance_model.load_model(
                project_dir
                / "data"
                / "interim"
                / "tree_experiments"
                / "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
                / "xgboost_model.json"
            )
        else:
            importance_model = train_xgb_for_importance(
                features, target, months, train_end
            )
        _, selection = select_correlated_features(
            features,
            feature_columns,
            months,
            train_end,
            importance_model,
        )
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        selections[train_end] = selection
        del importance_model
        gc.collect()

    inner_corr_indices = np.asarray(
        [column_to_index[name] for name in selections[39]["kept_features_in_source_order"]],
        dtype=np.int64,
    )
    formal_corr_indices = np.asarray(
        [column_to_index[name] for name in selections[49]["kept_features_in_source_order"]],
        dtype=np.int64,
    )
    all_indices = np.arange(len(feature_columns), dtype=np.int64)
    device = torch.device("cuda")
    corr_details = run_tabm_variant(
        "corrprune",
        features,
        target,
        months,
        inner_corr_indices,
        formal_corr_indices,
        "mse",
        device,
        run_dir,
    )
    cosine_details = run_tabm_variant(
        "cosine",
        features,
        target,
        months,
        all_indices,
        all_indices,
        "cosine",
        device,
        run_dir,
    )
    lightgbm_details = run_lightgbm(
        features, target, months, feature_columns, run_dir
    )
    del features
    gc.collect()

    b001_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
        / "validation_predictions.feather"
    )
    b001 = pd.read_feather(b001_path)
    if not np.array_equal(
        validation_rows["sample_id"].to_numpy(), b001["sample_id"].to_numpy()
    ):
        raise AssertionError("B001 validation rows are not aligned.")
    corr = pd.read_feather(run_dir / "corrprune_predictions.feather")
    cosine_frame = pd.read_feather(run_dir / "cosine_predictions.feather")
    lightgbm = pd.read_feather(run_dir / "lightgbm_predictions.feather")
    hgb_all = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-008-009"
        / "internal_forward_predictions.feather"
    )
    hgb = hgb_all.loc[
        (hgb_all["fold"] == "train_0_49_valid_50_59")
        & (hgb_all["variant"] == "baseline")
    ].reset_index(drop=True)
    if not np.array_equal(
        validation_rows["sample_id"].to_numpy(), hgb["sample_id"].to_numpy()
    ):
        raise AssertionError("HGB validation rows are not aligned.")

    sources = {
        "original_mean": centered_unit(b001["tabm_mean"].to_numpy(dtype=np.float64)),
        "original_trim1": centered_unit(
            b001["tabm_trim1"].to_numpy(dtype=np.float64)
        ),
        "corrprune_mean": unit(corr["mean_prediction"].to_numpy(dtype=np.float64)),
        "corrprune_trim1": unit(
            corr["trim1_prediction"].to_numpy(dtype=np.float64)
        ),
        "cosine_mean": unit(
            cosine_frame["mean_prediction"].to_numpy(dtype=np.float64)
        ),
        "cosine_trim1": unit(
            cosine_frame["trim1_prediction"].to_numpy(dtype=np.float64)
        ),
        "xgboost": centered_unit(b001["xgboost"].to_numpy(dtype=np.float64)),
        "lightgbm": centered_unit(lightgbm["prediction"].to_numpy(dtype=np.float64)),
        "histgb": unit(hgb["prediction"].to_numpy(dtype=np.float64)),
    }

    def six_source(
        weights: dict[str, float], original: str = "trim1", cosine_agg: str = "trim1"
    ) -> np.ndarray:
        return (
            weights["original"] * sources[f"original_{original}"]
            + weights["corrprune"] * sources["corrprune_mean"]
            + weights["cosine"] * sources[f"cosine_{cosine_agg}"]
            + weights["xgboost"] * sources["xgboost"]
            + weights["lightgbm"] * sources["lightgbm"]
            + weights["histgb"] * sources["histgb"]
        )

    b005_weights = {
        "original": 0.35,
        "corrprune": 0.20,
        "cosine": 0.15,
        "xgboost": 0.10,
        "lightgbm": 0.10,
        "histgb": 0.10,
    }
    maximin_weights = {
        "original": 0.3525,
        "corrprune": 0.20,
        "cosine": 0.1525,
        "xgboost": 0.0975,
        "lightgbm": 0.0975,
        "histgb": 0.10,
    }
    predictions = {
        "b001_mean": b001["b001_mean"].to_numpy(dtype=np.float64),
        "b001_trim1": b001["b001_trim1"].to_numpy(dtype=np.float64),
        "b005_exact": six_source(b005_weights),
        "maximin_exact": six_source(maximin_weights),
        "maximin_original_mean": six_source(maximin_weights, original="mean"),
        "maximin_cosine_mean": six_source(maximin_weights, cosine_agg="mean"),
        "maximin_both_mean": six_source(
            maximin_weights, original="mean", cosine_agg="mean"
        ),
    }
    validation_target = validation_rows["target"].to_numpy(dtype=np.float64)
    validation_months = validation_rows["month"].to_numpy()
    metrics = {
        name: evaluate(validation_target, prediction, validation_months)
        for name, prediction in predictions.items()
    }
    source_metrics = {
        name: evaluate(validation_target, prediction, validation_months)
        for name, prediction in sources.items()
    }
    output = validation_rows.copy()
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_feather(run_dir / "validation_predictions.feather")
    monthly = pd.DataFrame(
        {
            "month": list(range(FORMAL_VALID_START, FORMAL_VALID_END + 1)),
            **{
                name: [row["cosine"] for row in values["monthly"]]
                for name, values in metrics.items()
            },
        }
    )
    monthly.to_csv(run_dir / "monthly_cosine.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "candidate": name,
                **{key: value for key, value in result.items() if key != "monthly"},
                "change_vs_b001_mean": result["overall"]
                - metrics["b001_mean"]["overall"],
            }
            for name, result in metrics.items()
        ]
    ).sort_values("overall", ascending=False)
    summary.to_csv(run_dir / "candidate_summary.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": {
            "inner_train": "0-39",
            "inner_validation": "40-49",
            "formal_train": "0-49",
            "formal_validation": "50-59",
            "correlation_selection": "recomputed on each training fold only",
            "no_overlap": True,
        },
        "tabm_corrprune": corr_details,
        "tabm_cosine": cosine_details,
        "lightgbm": lightgbm_details,
        "metrics": metrics,
        "source_metrics": source_metrics,
        "b005_weights": b005_weights,
        "maximin_weights": maximin_weights,
        "prediction_path": str(run_dir / "validation_predictions.feather"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(monthly.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
