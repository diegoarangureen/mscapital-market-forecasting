"""Paired seed-137 confirmation of static versus joint GRU319 on two folds."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch

import exp_gru_003_strong_joint as experiment
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import QuantileBatchPreprocessor, fit_quantile_knots
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS


EXPERIMENT_ID = "EXP-GRU-005-JOINT-SEED137"
SEED = 137
EPOCHS = 6
FOLDS = {
    "train049_valid5059": (49, 50, 59),
    "train059_valid6070": (59, 60, 70),
}


def score_subsets(frame: pd.DataFrame, prediction_column: str, baseline_column: str) -> dict:
    target = frame["target"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    baseline = frame[baseline_column].to_numpy(dtype=np.float64)
    months = frame["month"].to_numpy()

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denominator) if denominator else 0.0

    subsets = {
        "overall": np.ones(len(frame), dtype=bool),
        "without_month_66": months != 66,
        "months_62_70_without_66": (months >= 62) & (months != 66),
        "months_67_70": months >= 67,
    }
    result = {}
    for name, mask in subsets.items():
        if not bool(mask.any()):
            continue
        score = cosine(target[mask], prediction[mask])
        base_score = cosine(target[mask], baseline[mask])
        result[name] = score
        result[name + "_baseline"] = base_score
        result[name + "_delta"] = score - base_score
    return result


def run_fold(
    *,
    project_dir: Path,
    root_dir: Path,
    fold_name: str,
    train_end_month: int,
    valid_start_month: int,
    valid_end_month: int,
    cache: np.memmap,
    labels: pd.DataFrame,
    static_features: np.ndarray,
    feature_columns: list[str],
    raw_means: np.ndarray,
    raw_scales: np.ndarray,
    device: torch.device,
) -> dict:
    fold_dir = root_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    experiment.SEED = SEED
    experiment.EPOCHS = EPOCHS
    experiment.EVALUATION_EPOCHS = {EPOCHS}
    experiment.FOLD_NAME = fold_name
    experiment.VALID_START_MONTH = valid_start_month
    experiment.VALID_END_MONTH = valid_end_month

    months = labels["month"].to_numpy()
    train_end = int(np.searchsorted(months, train_end_month + 1, side="left"))
    valid_start = int(np.searchsorted(months, valid_start_month, side="left"))
    valid_end = int(np.searchsorted(months, valid_end_month + 1, side="left"))
    if train_end != valid_start:
        raise AssertionError("Expected contiguous forward split.")
    validation_rows = labels.iloc[valid_start:valid_end].copy()
    baseline = experiment.load_baseline(
        project_dir, validation_rows["sample_id"].to_numpy()
    )

    fold_means, fold_scales = fit_fold_standardization(
        cache, train_end, fold_dir / "fold_standardization.npz"
    )
    builder = FullSequenceBatchBuilder(
        cache, raw_means, raw_scales, fold_means, fold_scales, device
    )
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[:train_end].mean(dtype=np.float64))
    target_scale = float(target[:train_end].std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_scale).astype(np.float32)

    train_indices = np.arange(train_end, dtype=np.int64)
    preprocessing_path = fold_dir / "quantile_preprocessing.npz"
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
    else:
        knots, medians, missing_columns = fit_quantile_knots(
            static_features, train_indices, SEED
        )
        np.savez_compressed(
            preprocessing_path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray(feature_columns),
        )
    preprocessor = QuantileBatchPreprocessor(knots, medians, missing_columns, device)
    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)
    results = {}
    for name, use_sequence in (("static_control", False), ("joint_gru319", True)):
        experiment.seed_everything(SEED)
        model = experiment.JointRegressor(
            input_size, preprocessor.output_dimension, use_sequence
        ).to(device)
        experiment.check_padding_invariance(model, input_size, device)
        results[name] = experiment.train_candidate(
            name=name,
            model=model,
            builder=builder,
            target_scaled=target_scaled,
            target_scale=target_scale,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            validation_rows=validation_rows,
            baseline=baseline,
            run_dir=fold_dir,
            static_features=static_features,
            static_preprocessor=preprocessor,
            device=device,
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()

    static_frame = pd.read_feather(
        fold_dir / "static_control" / "validation_predictions_epoch06.feather"
    )
    joint_frame = pd.read_feather(
        fold_dir / "joint_gru319" / "validation_predictions_epoch06.feather"
    )
    for column in ("sample_id", "month", "target", "b001_trim1"):
        if not np.array_equal(static_frame[column].to_numpy(), joint_frame[column].to_numpy()):
            raise AssertionError(f"Static/joint {column} alignment differs.")
    static_metrics = score_subsets(static_frame, "blend90", "b001_trim1")
    joint_metrics = score_subsets(joint_frame, "blend90", "b001_trim1")
    comparison = {
        key: joint_metrics[key] - static_metrics[key]
        for key in joint_metrics
        if key.endswith("_delta")
    }
    result = {
        "fold": fold_name,
        "seed": SEED,
        "train_months": f"0-{train_end_month}",
        "validation_months": f"{valid_start_month}-{valid_end_month}",
        "models": results,
        "static_blend_metrics": static_metrics,
        "joint_blend_metrics": joint_metrics,
        "joint_minus_static_delta": comparison,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "fold": fold_name,
        "static": static_metrics,
        "joint": joint_metrics,
        "joint_minus_static": comparison,
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    device = torch.device("cuda")
    project_dir = Path(__file__).resolve().parents[1]
    root_dir = (
        project_dir / "data" / "interim" / "sequence_experiments" / EXPERIMENT_ID
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    if cache.shape != (len(labels), 14, 200):
        raise AssertionError("Sequence cache and labels differ.")
    experiment.check_cache_layout(cache)
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    static_features, feature_columns = experiment.load_relative319(project_dir, labels)
    if len(feature_columns) != 319 or feature_columns[-len(RELATIVE_COLUMNS):] != RELATIVE_COLUMNS:
        raise AssertionError("Relative319 feature definition differs.")

    results = {}
    for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
        results[fold_name] = run_fold(
            project_dir=project_dir,
            root_dir=root_dir,
            fold_name=fold_name,
            train_end_month=train_end,
            valid_start_month=valid_start,
            valid_end_month=valid_end,
            cache=cache,
            labels=labels,
            static_features=static_features,
            feature_columns=feature_columns,
            raw_means=raw_means,
            raw_scales=raw_scales,
            device=device,
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "seed": SEED,
        "epochs": EPOCHS,
        "single_change_from_seed42": "paired random seed change",
        "folds": results,
    }
    (root_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

