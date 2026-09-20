"""Build target-free frequency and path-shape features from the 3-second market grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 5_000
MARKET_LEN = 200
MARKET_DIM = 11
FEATURE_COLUMNS = [
    "mid_return_autocorr_1_full",
    "mid_return_autocorr_2_full",
    "mid_return_autocorr_5_full",
    "mid_return_autocorr_10_full",
    "mid_return_autocorr_1_60",
    "mid_return_autocorr_2_60",
    "mid_return_autocorr_5_60",
    "mid_variance_ratio_2",
    "mid_variance_ratio_5",
    "mid_variance_ratio_10",
    "mid_spectral_low_fraction",
    "mid_spectral_mid_fraction",
    "mid_spectral_high_fraction",
    "mid_spectral_entropy",
    "mid_dominant_frequency",
    "mid_path_efficiency_full",
    "mid_path_efficiency_60",
    "mid_sign_persistence",
    "mid_reversal_magnitude_fraction",
    "mid_jump_concentration_top5",
    "mid_max_drawdown_log",
    "mid_max_runup_log",
    "mid_rv_ratio_20_to_60",
    "mid_rv_ratio_60_to_full",
]


def discover_grid(split: str) -> tuple[Path, Path]:
    file_names = (
        f"{split}_v2_market_{MARKET_LEN}x{MARKET_DIM}.mmap",
        f"{split}_v2_market_count_{MARKET_LEN}.mmap",
    )
    candidates = [
        PROJECT / "data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid",
        PROJECT / "data/interim/kaggle_outputs/multistream_grid_cache_v2/mscapital_multistream_grid_cache_v2",
    ]
    for directory in candidates:
        paths = directory / file_names[0], directory / file_names[1]
        if paths[0].exists() and paths[1].exists():
            return paths
    raise FileNotFoundError(f"No complete {split} market grid cache was found")


def forward_fill(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(values.shape[1], dtype=np.int16)[None, :]
    last_seen = np.maximum.accumulate(np.where(valid, positions, -1), axis=1)
    any_valid = valid.any(axis=1)
    first_seen = np.argmax(valid, axis=1)
    leading = last_seen < 0
    last_seen[leading] = np.broadcast_to(first_seen[:, None], last_seen.shape)[leading]
    filled = np.take_along_axis(values, last_seen, axis=1)
    filled[~any_valid] = 0.0
    return filled, any_valid


def row_autocorr(returns: np.ndarray, lag: int) -> np.ndarray:
    left = returns[:, :-lag]
    right = returns[:, lag:]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = (left * right).sum(axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.square(left).sum(axis=1, dtype=np.float64)
        * np.square(right).sum(axis=1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(len(returns), dtype=np.float64),
        where=denominator > 1e-16,
    )


def variance_ratio(returns: np.ndarray, horizon: int) -> np.ndarray:
    width = (returns.shape[1] // horizon) * horizon
    blocks = returns[:, -width:].reshape(len(returns), -1, horizon).sum(axis=2)
    base_variance = returns.var(axis=1, dtype=np.float64)
    block_variance = blocks.var(axis=1, dtype=np.float64)
    denominator = horizon * base_variance
    return np.divide(
        block_variance,
        denominator,
        out=np.zeros(len(returns), dtype=np.float64),
        where=denominator > 1e-16,
    )


def path_efficiency(returns: np.ndarray) -> np.ndarray:
    traveled = np.abs(returns).sum(axis=1, dtype=np.float64)
    displacement = np.abs(returns.sum(axis=1, dtype=np.float64))
    return np.divide(
        displacement,
        traveled,
        out=np.zeros(len(returns), dtype=np.float64),
        where=traveled > 1e-16,
    )


def build_features(raw_market: np.ndarray, count: np.ndarray) -> np.ndarray:
    mid_relative = raw_market[:, :, 0].astype(np.float32)
    valid = (count > 0) & np.isfinite(mid_relative) & (mid_relative > -0.9)
    mid_relative, any_valid = forward_fill(mid_relative, valid)
    log_mid = np.log1p(np.clip(mid_relative, -0.9, 10.0))
    returns = np.diff(log_mid, axis=1)
    recent60 = returns[:, -60:]
    recent20 = returns[:, -20:]
    output = np.zeros((len(raw_market), len(FEATURE_COLUMNS)), dtype=np.float32)

    for column, lag in enumerate((1, 2, 5, 10)):
        output[:, column] = row_autocorr(returns, lag)
    for column, lag in zip((4, 5, 6), (1, 2, 5)):
        output[:, column] = row_autocorr(recent60, lag)
    for column, horizon in zip((7, 8, 9), (2, 5, 10)):
        output[:, column] = np.clip(variance_ratio(returns, horizon), 0.0, 10.0)

    centered = returns - returns.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    spectrum[:, 0] = 0.0
    total_energy = spectrum.sum(axis=1, dtype=np.float64)
    low_energy = spectrum[:, 1:5].sum(axis=1, dtype=np.float64)
    mid_energy = spectrum[:, 5:17].sum(axis=1, dtype=np.float64)
    high_energy = spectrum[:, 17:].sum(axis=1, dtype=np.float64)
    for column, energy in zip((10, 11, 12), (low_energy, mid_energy, high_energy)):
        output[:, column] = np.divide(
            energy,
            total_energy,
            out=np.zeros(len(raw_market), dtype=np.float64),
            where=total_energy > 1e-16,
        )
    probabilities = np.divide(
        spectrum[:, 1:],
        total_energy[:, None],
        out=np.zeros_like(spectrum[:, 1:], dtype=np.float64),
        where=total_energy[:, None] > 1e-16,
    )
    output[:, 13] = np.divide(
        -(probabilities * np.log(probabilities + 1e-20)).sum(axis=1),
        np.log(probabilities.shape[1]),
    )
    output[:, 14] = (np.argmax(spectrum[:, 1:], axis=1) + 1) / (spectrum.shape[1] - 1)

    output[:, 15] = path_efficiency(returns)
    output[:, 16] = path_efficiency(recent60)
    signs = np.sign(returns)
    sign_pairs = signs[:, 1:] * signs[:, :-1]
    nonzero_pairs = (signs[:, 1:] != 0) & (signs[:, :-1] != 0)
    pair_count = nonzero_pairs.sum(axis=1)
    output[:, 17] = np.divide(
        np.where(nonzero_pairs, sign_pairs, 0.0).sum(axis=1),
        pair_count,
        out=np.zeros(len(raw_market), dtype=np.float64),
        where=pair_count > 0,
    )
    reversal = nonzero_pairs & (sign_pairs < 0)
    absolute_returns = np.abs(returns)
    output[:, 18] = np.divide(
        np.where(reversal, absolute_returns[:, 1:], 0.0).sum(axis=1),
        absolute_returns[:, 1:].sum(axis=1),
        out=np.zeros(len(raw_market), dtype=np.float64),
        where=absolute_returns[:, 1:].sum(axis=1) > 1e-16,
    )
    top5 = np.partition(absolute_returns, -5, axis=1)[:, -5:].sum(axis=1)
    total_absolute = absolute_returns.sum(axis=1)
    output[:, 19] = np.divide(
        top5,
        total_absolute,
        out=np.zeros(len(raw_market), dtype=np.float64),
        where=total_absolute > 1e-16,
    )
    running_max = np.maximum.accumulate(log_mid, axis=1)
    running_min = np.minimum.accumulate(log_mid, axis=1)
    output[:, 20] = np.max(running_max - log_mid, axis=1)
    output[:, 21] = np.max(log_mid - running_min, axis=1)
    rv_full = np.sqrt(np.square(returns).sum(axis=1, dtype=np.float64))
    rv60 = np.sqrt(np.square(recent60).sum(axis=1, dtype=np.float64))
    rv20 = np.sqrt(np.square(recent20).sum(axis=1, dtype=np.float64))
    output[:, 22] = np.divide(rv20, rv60, out=np.zeros(len(raw_market)), where=rv60 > 1e-16)
    output[:, 23] = np.divide(rv60, rv_full, out=np.zeros(len(raw_market)), where=rv_full > 1e-16)
    output[~any_valid] = 0.0
    return np.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.split == "train":
        id_frame = pd.read_feather(
            PROJECT / "data/raw/label.feather", columns=["sample_id"]
        )
    else:
        id_frame = pd.read_csv(
            PROJECT / "data/raw/submission.csv", usecols=["sample_id"]
        )
    sample_ids = id_frame["sample_id"].to_numpy()
    market_path, count_path = discover_grid(args.split)
    market = np.memmap(
        market_path, dtype=np.float16, mode="r",
        shape=(len(sample_ids), MARKET_LEN, MARKET_DIM),
    )
    count = np.memmap(
        count_path, dtype=np.float16, mode="r",
        shape=(len(sample_ids), MARKET_LEN),
    )
    row_count = min(args.limit, len(sample_ids)) if args.limit else len(sample_ids)
    output = np.empty((row_count, len(FEATURE_COLUMNS)), dtype=np.float32)
    for start in range(0, row_count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, row_count)
        output[start:end] = build_features(
            np.asarray(market[start:end], dtype=np.float32),
            np.asarray(count[start:end], dtype=np.float32),
        )
        if start % 50_000 == 0:
            print(f"{args.split} path features {end:,}/{row_count:,}", flush=True)
    if not np.isfinite(output).all():
        raise AssertionError("Path/spectral features contain nonfinite values")
    frame = pd.DataFrame(output, columns=FEATURE_COLUMNS)
    frame.insert(0, "sample_id", sample_ids[:row_count])
    suffix = f"_smoke{row_count}" if args.limit else ""
    destination = (
        PROJECT / "data/processed" / f"{args.split}_market_path_spectral_features{suffix}.feather"
    )
    frame.to_feather(destination)
    metadata = {
        "split": args.split,
        "rows": row_count,
        "features": FEATURE_COLUMNS,
        "source_market": str(market_path),
        "source_count": str(count_path),
        "output": str(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
