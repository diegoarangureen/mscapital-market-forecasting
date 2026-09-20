"""Strict forward OOF residual correction for TabM385-k32."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP-STACK-002-TABM-FORWARD-OOF-RESIDUAL"
RUN_DIR = PROJECT / "data/interim/tree_experiments" / EXPERIMENT_ID
OOF_ROOT = PROJECT / "data/interim/tree_experiments/EXP-TABM-041-FORWARD-OOF-BLOCKS"
LATE_ROOT = PROJECT / "data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66"
MARKET6 = [
    "mid_return_180", "mid_return_60", "mid_return_20",
    "realized_volatility_20", "realized_volatility_60", "mid_momentum_60",
]
OOF_FOLDS = [
    "train027_valid3035",
    "train033_valid3641",
    "train039_valid4247",
    "train045_valid4853",
    "train051_valid5459",
]
RESIDUAL_WEIGHT = 0.25


def cosine(target, prediction):
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def scores(target, prediction, months):
    return {
        "overall": cosine(target, prediction),
        "monthly": {
            str(int(month)): cosine(target[months == month], prediction[months == month])
            for month in np.unique(months)
        },
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_feather(
        PROJECT / "data/raw/label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    sample_ids = labels["sample_id"].to_numpy()
    months = labels["month"].to_numpy(np.int16)
    target = labels["target"].to_numpy(np.float64)

    oof_parts = []
    for fold in OOF_FOLDS:
        path = OOF_ROOT / fold / "validation_predictions.feather"
        if not path.exists():
            raise FileNotFoundError(f"Missing forward OOF block: {path}")
        oof_parts.append(pd.read_feather(path))
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if oof["sample_id"].duplicated().any():
        raise AssertionError("Forward OOF sample IDs overlap.")
    expected_oof = labels.loc[labels["month"].between(30, 59), ["sample_id", "month", "target"]]
    if not np.array_equal(oof["sample_id"].to_numpy(), expected_oof["sample_id"].to_numpy()):
        raise AssertionError("Forward OOF does not cover every sample in months 30-59 exactly once.")
    if not np.array_equal(oof["month"].to_numpy(), expected_oof["month"].to_numpy()):
        raise AssertionError("Forward OOF months do not align.")
    oof_indices = oof["sample_id"].to_numpy(dtype=np.int64)
    oof_prediction = oof["prediction"].to_numpy(np.float64)
    oof_target = oof["target"].to_numpy(np.float64)

    late = pd.read_feather(LATE_ROOT / "validation_predictions.feather")
    late = late.loc[late["month"].ne(66)].sort_values("sample_id").reset_index(drop=True)
    late_indices = late["sample_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids[late_indices], late_indices):
        raise AssertionError("This dataset is expected to use dense ordered sample IDs.")
    late_prediction = late["candidate"].to_numpy(np.float64)
    late_target = late["target"].to_numpy(np.float64)
    late_months = late["month"].to_numpy(np.int16)

    cache = PROJECT / "data/interim/our379_reference_cache"
    base379 = np.load(cache / "features.npy", mmap_mode="r")
    if base379.shape != (len(sample_ids), 379):
        raise AssertionError("our379 cache shape does not align.")
    if not np.array_equal(sample_ids, np.arange(len(sample_ids), dtype=sample_ids.dtype)):
        raise AssertionError("Labels must be in dense sample_id order.")
    state = pd.read_feather(
        PROJECT / "data/processed/train_market_microstructure_features.feather",
        columns=["sample_id", *MARKET6],
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(state["sample_id"].to_numpy(), sample_ids):
        raise AssertionError("Market6 IDs do not align.")
    market6 = state[MARKET6].to_numpy(np.float32, copy=True)
    del state
    gc.collect()

    x_oof = np.concatenate(
        [
            np.asarray(base379[oof_indices], dtype=np.float32),
            market6[oof_indices],
            oof_prediction[:, None].astype(np.float32),
        ],
        axis=1,
    )
    x_late = np.concatenate(
        [
            np.asarray(base379[late_indices], dtype=np.float32),
            market6[late_indices],
            late_prediction[:, None].astype(np.float32),
        ],
        axis=1,
    )
    del base379, market6
    gc.collect()
    if x_oof.shape[1] != 386 or x_late.shape[1] != 386:
        raise AssertionError("Expected our385 plus one TabM prediction feature.")

    residual_target = oof_target - oof_prediction
    target_scale = float(np.std(oof_target))
    parameters = {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "tree_learner": "serial",
        "num_threads": 2,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 1000,
        "lambda_l2": 100.0,
        "lambda_l1": 0.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "max_bin": 127,
        "seed": 42,
        "verbosity": -1,
        "force_col_wise": True,
    }

    print(f"training residual tree rows={len(x_oof):,}, features={x_oof.shape[1]}", flush=True)
    residual_model = lgb.train(
        parameters,
        lgb.Dataset(x_oof, label=residual_target / target_scale, free_raw_data=False),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    residual_prediction = (
        np.asarray(residual_model.predict(x_late), dtype=np.float64) * target_scale
    )
    corrected_prediction = late_prediction + RESIDUAL_WEIGHT * residual_prediction
    del residual_model
    gc.collect()

    print("training equal-capacity direct-y control tree", flush=True)
    direct_model = lgb.train(
        parameters,
        lgb.Dataset(x_oof, label=oof_target / target_scale, free_raw_data=False),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    direct_prediction = (
        np.asarray(direct_model.predict(x_late), dtype=np.float64) * target_scale
    )
    direct_blend = (1.0 - RESIDUAL_WEIGHT) * late_prediction + RESIDUAL_WEIGHT * direct_prediction

    baseline = scores(late_target, late_prediction, late_months)
    residual_result = scores(late_target, corrected_prediction, late_months)
    direct_result = scores(late_target, direct_blend, late_months)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "meta_train": "strict forward TabM OOF months30-59 in five 6-month blocks, 2-month purge",
        "outer_validation": "TabM trained0-59, validation62-70 excluding66",
        "features": "our385 plus TabM prediction",
        "tree_parameters": parameters,
        "rounds": 300,
        "residual_weight": RESIDUAL_WEIGHT,
        "oof_rows": int(len(oof)),
        "validation_rows": int(len(late)),
        "oof_tabm_cosine": cosine(oof_target, oof_prediction),
        "baseline": baseline,
        "residual_correction": residual_result,
        "direct_y_control": direct_result,
        "residual_delta_vs_baseline": residual_result["overall"] - baseline["overall"],
        "direct_delta_vs_baseline": direct_result["overall"] - baseline["overall"],
        "residual_minus_direct": residual_result["overall"] - direct_result["overall"],
        "pass_threshold": 0.0007,
        "passed": bool(
            residual_result["overall"] - baseline["overall"] >= 0.0007
            and residual_result["overall"] > direct_result["overall"]
        ),
    }
    pd.DataFrame(
        {
            "sample_id": late_indices,
            "month": late_months,
            "target": late_target,
            "tabm_baseline": late_prediction,
            "predicted_residual": residual_prediction,
            "residual_corrected": corrected_prediction,
            "direct_tree": direct_prediction,
            "direct_blend": direct_blend,
        }
    ).to_feather(RUN_DIR / "validation_predictions.feather")
    (RUN_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
