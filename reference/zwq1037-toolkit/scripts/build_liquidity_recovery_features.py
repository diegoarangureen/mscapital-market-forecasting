"""Build target-free post-shock liquidity recovery features from the market cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from build_train_market_microstructure_features import CHANNELS, unscale

BATCH_SIZE = 2_500
FEATURE_COLUMNS = [
    "spread_shock_count_60",
    "spread_recovery_fraction_6",
    "spread_recovery_fraction_15",
    "depth_shock_count_60",
    "depth_recovery_fraction_6",
    "depth_recovery_fraction_15",
    "price_reversion_fraction_15",
    "liquidity_recovery_coverage_60",
]


def load_statistics(cache_dir: Path):
    statistics = {}
    for name in CHANNELS:
        path = cache_dir / "statistics" / f"{name}.json"
        if path.exists():
            statistics[name] = json.loads(path.read_text(encoding="utf-8"))
    return statistics


def series_from_block(block, statistics):
    row_mask = block[:, CHANNELS["row_mask"], :] > 0.5
    seconds = block[:, CHANNELS["seconds_before_predict"], :].astype(np.float32) * 600.0
    ask1 = unscale(block, "ask_price_1", statistics)
    bid1 = unscale(block, "bid_price_1", statistics)
    askv1 = unscale(block, "ask_volume_1", statistics)
    bidv1 = unscale(block, "bid_volume_1", statistics)
    askv2 = unscale(block, "ask_volume_2", statistics)
    bidv2 = unscale(block, "bid_volume_2", statistics)

    mid = (ask1 + bid1) * 0.5
    valid_price = row_mask & np.isfinite(mid) & (mid > 0)
    relative_spread = np.divide(
        ask1 - bid1,
        mid,
        out=np.full_like(mid, np.nan),
        where=valid_price,
    )
    valid_spread = valid_price & np.isfinite(relative_spread) & (relative_spread > 0)
    log_spread = np.log(np.clip(relative_spread, 1.0e-8, None))

    depth = askv1 + bidv1 + askv2 + bidv2
    valid_depth = row_mask & np.isfinite(depth) & (depth > 0)
    log_depth = np.log1p(np.clip(depth, 0, None))
    log_mid = np.log(np.clip(mid, 1.0e-12, None))
    return seconds, row_mask, log_spread, valid_spread, log_depth, valid_depth, log_mid, valid_price


def changes_from_series(series):
    seconds, row_mask, log_spread, valid_spread, log_depth, valid_depth, log_mid, valid_price = series
    eligible = (
        row_mask[:, 1:]
        & row_mask[:, :-1]
        & (seconds[:, 1:] >= 15.0)
        & (seconds[:, 1:] <= 60.0)
    )
    spread_valid = eligible & valid_spread[:, 1:] & valid_spread[:, :-1]
    depth_valid = eligible & valid_depth[:, 1:] & valid_depth[:, :-1]
    price_valid = eligible & valid_price[:, 1:] & valid_price[:, :-1]
    spread_change = log_spread[:, 1:] - log_spread[:, :-1]
    depth_drop = log_depth[:, :-1] - log_depth[:, 1:]
    price_change = log_mid[:, 1:] - log_mid[:, :-1]
    return eligible, spread_change, spread_valid, depth_drop, depth_valid, price_change, price_valid


def fit_thresholds(cache, statistics, train_end_count):
    blocks = 12
    block_size = 5_000
    starts = np.linspace(
        0, max(train_end_count - block_size, 0), blocks, dtype=np.int64
    )
    collected = {"spread": [], "depth": [], "price": []}
    for start in starts:
        end = min(int(start) + block_size, train_end_count)
        changes = changes_from_series(series_from_block(cache[int(start):end], statistics))
        _, spread, spread_valid, depth, depth_valid, price, price_valid = changes
        collected["spread"].append(spread[spread_valid & (spread > 0)].astype(np.float32))
        collected["depth"].append(depth[depth_valid & (depth > 0)].astype(np.float32))
        collected["price"].append(np.abs(price[price_valid]).astype(np.float32))
    thresholds = {
        name: float(np.quantile(np.concatenate(values), 0.95))
        for name, values in collected.items()
    }
    if not all(np.isfinite(list(thresholds.values()))):
        raise AssertionError(f"Nonfinite thresholds: {thresholds}")
    return thresholds


def gather_future(values, valid, event_index, offset):
    future_index = np.minimum(event_index + offset, values.shape[1] - 1)
    gathered = np.take_along_axis(values, future_index[:, None], axis=1)[:, 0]
    is_valid = np.take_along_axis(valid, future_index[:, None], axis=1)[:, 0]
    is_valid &= event_index + offset < values.shape[1]
    return gathered, is_valid


def build_features(block, statistics, thresholds):
    series = series_from_block(block, statistics)
    seconds, row_mask, log_spread, valid_spread, log_depth, valid_depth, log_mid, valid_price = series
    eligible, spread, spread_valid, depth, depth_valid, price, price_valid = changes_from_series(series)
    n = len(block)
    output = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)

    spread_shocks = spread_valid & (spread > thresholds["spread"])
    spread_metric = np.where(spread_valid, spread, -np.inf)
    spread_arg = np.argmax(spread_metric, axis=1) + 1
    spread_peak = np.take_along_axis(spread, (spread_arg - 1)[:, None], axis=1)[:, 0]
    has_spread = spread_shocks.any(axis=1)
    output[:, 0] = spread_shocks.sum(axis=1)
    current_spread = np.take_along_axis(log_spread, spread_arg[:, None], axis=1)[:, 0]
    for column, offset in ((1, 2), (2, 5)):
        future, future_valid = gather_future(log_spread, valid_spread, spread_arg, offset)
        good = has_spread & future_valid & np.isfinite(spread_peak) & (spread_peak > 1.0e-12)
        values = np.zeros(n, dtype=np.float64)
        values[good] = (current_spread[good] - future[good]) / spread_peak[good]
        output[:, column] = np.clip(values, -3.0, 3.0)

    depth_shocks = depth_valid & (depth > thresholds["depth"])
    depth_metric = np.where(depth_valid, depth, -np.inf)
    depth_arg = np.argmax(depth_metric, axis=1) + 1
    depth_peak = np.take_along_axis(depth, (depth_arg - 1)[:, None], axis=1)[:, 0]
    has_depth = depth_shocks.any(axis=1)
    output[:, 3] = depth_shocks.sum(axis=1)
    current_depth = np.take_along_axis(log_depth, depth_arg[:, None], axis=1)[:, 0]
    for column, offset in ((4, 2), (5, 5)):
        future, future_valid = gather_future(log_depth, valid_depth, depth_arg, offset)
        good = has_depth & future_valid & np.isfinite(depth_peak) & (depth_peak > 1.0e-12)
        values = np.zeros(n, dtype=np.float64)
        values[good] = (future[good] - current_depth[good]) / depth_peak[good]
        output[:, column] = np.clip(values, -3.0, 3.0)

    price_shocks = price_valid & (np.abs(price) > thresholds["price"])
    price_metric = np.where(price_valid, np.abs(price), -np.inf)
    price_arg = np.argmax(price_metric, axis=1) + 1
    price_peak = np.take_along_axis(price, (price_arg - 1)[:, None], axis=1)[:, 0]
    has_price = price_shocks.any(axis=1)
    current_mid = np.take_along_axis(log_mid, price_arg[:, None], axis=1)[:, 0]
    future_mid, future_mid_valid = gather_future(log_mid, valid_price, price_arg, 5)
    good = has_price & future_mid_valid & np.isfinite(price_peak) & (np.abs(price_peak) > 1.0e-12)
    reversion = np.zeros(n, dtype=np.float64)
    reversion[good] = (
        -np.sign(price_peak[good])
        * (future_mid[good] - current_mid[good])
        / np.abs(price_peak[good])
    )
    output[:, 6] = np.clip(reversion, -3.0, 3.0)
    output[:, 7] = np.minimum(eligible.sum(axis=1) / 16.0, 1.0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cache_dir = PROJECT / "data" / "processed" / f"{args.split}_sequence_cache_v1"
    train_cache_dir = PROJECT / "data" / "processed" / "train_sequence_cache_v1"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    statistics = load_statistics(train_cache_dir)
    if args.split == "train":
        ids = pd.read_feather(PROJECT / "data" / "raw" / "label.feather", columns=["sample_id", "month"])
        sample_ids = ids["sample_id"].to_numpy()
        train_end_count = int((ids["month"].to_numpy() <= 59).sum())
    else:
        ids = pd.read_csv(PROJECT / "data" / "raw" / "submission.csv", usecols=["sample_id"])
        sample_ids = ids["sample_id"].to_numpy()
        train_ids = pd.read_feather(PROJECT / "data" / "raw" / "label.feather", columns=["month"])
        train_end_count = int((train_ids["month"].to_numpy() <= 59).sum())
    if len(sample_ids) != cache.shape[0]:
        raise AssertionError("Cache and sample ID row counts differ.")
    thresholds_path = PROJECT / "data" / "processed" / "liquidity_recovery_thresholds.json"
    train_cache = np.load(train_cache_dir / "sequences.npy", mmap_mode="r")
    if args.split == "train" or not thresholds_path.exists():
        thresholds = fit_thresholds(train_cache, statistics, train_end_count)
        if not args.limit:
            thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    else:
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    row_count = min(args.limit, len(sample_ids)) if args.limit else len(sample_ids)
    output = np.empty((row_count, len(FEATURE_COLUMNS)), dtype=np.float32)
    for start in range(0, row_count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, row_count)
        output[start:end] = build_features(cache[start:end], statistics, thresholds)
        if start % 50_000 == 0:
            print(f"{args.split} recovery features {end:,}/{row_count:,}", flush=True)
    if not np.isfinite(output).all():
        raise AssertionError("Recovery features contain nonfinite values.")
    frame = pd.DataFrame(output, columns=FEATURE_COLUMNS)
    frame.insert(0, "sample_id", sample_ids[:row_count])
    suffix = f"_smoke{row_count}" if args.limit else ""
    destination = PROJECT / "data" / "processed" / f"{args.split}_liquidity_recovery_features{suffix}.feather"
    frame.to_feather(destination)
    metadata = {
        "split": args.split,
        "rows": row_count,
        "features": FEATURE_COLUMNS,
        "thresholds": thresholds,
        "threshold_fit_rows": f"train samples 0-{train_end_count - 1}",
        "source": str(cache_dir / "sequences.npy"),
        "output": str(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

