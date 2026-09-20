"""Build four time-weighted L1-versus-L2 order-book shape features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_market_last15_baseline_features import safe_write
from build_train_market_microstructure_features import CHANNELS, unscale


BATCH_SIZE = 10_000
IMBALANCE_DEAD_ZONE = 0.05
FEATURE_COLUMNS = [
    "near_depth_share_60",
    "imbalance_gap_l1_l2_60",
    "imbalance_disagreement_ratio_60",
    "near_depth_share_change_15_vs_60",
]


def interval_weights(
    seconds: np.ndarray, row_mask: np.ndarray, window_seconds: float
) -> np.ndarray:
    """Return the observed duration represented by every snapshot in a window."""

    next_seconds = np.zeros_like(seconds)
    next_seconds[:, :-1] = seconds[:, 1:]
    weights = np.clip(
        np.minimum(seconds, window_seconds) - np.maximum(next_seconds, 0.0),
        0.0,
        None,
    )
    return np.where(row_mask, weights, 0.0)


def weighted_mean(
    values: np.ndarray, valid: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    selected_weights = np.where(valid, weights, 0.0)
    denominator = selected_weights.sum(axis=1, dtype=np.float64)
    numerator = (np.where(valid, values, 0.0) * selected_weights).sum(
        axis=1, dtype=np.float64
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )


def build_batch(cache_block: np.ndarray, statistics: dict) -> dict[str, np.ndarray]:
    row_mask = cache_block[:, CHANNELS["row_mask"], :] > 0.5
    seconds = (
        cache_block[:, CHANNELS["seconds_before_predict"], :].astype(np.float64)
        * 600.0
    )
    weights_60 = interval_weights(seconds, row_mask, 60.0)
    weights_15 = interval_weights(seconds, row_mask, 15.0)

    ask_volume_1 = np.maximum(unscale(cache_block, "ask_volume_1", statistics), 0.0)
    bid_volume_1 = np.maximum(unscale(cache_block, "bid_volume_1", statistics), 0.0)
    ask_volume_2 = np.maximum(unscale(cache_block, "ask_volume_2", statistics), 0.0)
    bid_volume_2 = np.maximum(unscale(cache_block, "bid_volume_2", statistics), 0.0)
    depth_1 = ask_volume_1 + bid_volume_1
    depth_2 = ask_volume_2 + bid_volume_2
    valid = (
        row_mask
        & (ask_volume_1 > 0)
        & (bid_volume_1 > 0)
        & (ask_volume_2 > 0)
        & (bid_volume_2 > 0)
    )
    near_depth_share = np.divide(
        depth_1,
        depth_1 + depth_2,
        out=np.zeros_like(depth_1),
        where=valid,
    )
    imbalance_1 = np.divide(
        bid_volume_1 - ask_volume_1,
        depth_1,
        out=np.zeros_like(depth_1),
        where=valid,
    )
    imbalance_2 = np.divide(
        bid_volume_2 - ask_volume_2,
        depth_2,
        out=np.zeros_like(depth_2),
        where=valid,
    )
    imbalance_gap = imbalance_1 - imbalance_2
    disagreement = (
        valid
        & (np.abs(imbalance_1) > IMBALANCE_DEAD_ZONE)
        & (np.abs(imbalance_2) > IMBALANCE_DEAD_ZONE)
        & (np.signbit(imbalance_1) != np.signbit(imbalance_2))
    )

    near_share_60 = weighted_mean(near_depth_share, valid, weights_60)
    near_share_15 = weighted_mean(near_depth_share, valid, weights_15)
    valid_seconds_60 = np.where(valid, weights_60, 0.0).sum(
        axis=1, dtype=np.float64
    )
    disagreement_seconds = np.where(disagreement, weights_60, 0.0).sum(
        axis=1, dtype=np.float64
    )
    disagreement_ratio = np.divide(
        disagreement_seconds,
        valid_seconds_60,
        out=np.full(len(valid), np.nan),
        where=valid_seconds_60 > 0,
    )
    return {
        "near_depth_share_60": near_share_60,
        "imbalance_gap_l1_l2_60": weighted_mean(imbalance_gap, valid, weights_60),
        "imbalance_disagreement_ratio_60": disagreement_ratio,
        "near_depth_share_change_15_vs_60": near_share_15 - near_share_60,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    cache_dir = processed_dir / f"{arguments.split}_sequence_cache_v1"
    train_cache_dir = processed_dir / "train_sequence_cache_v1"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    sample_ids = pd.read_feather(
        processed_dir / f"{arguments.split}_market_features.feather",
        columns=["sample_id"],
    )["sample_id"].to_numpy(dtype=np.int32)
    if len(sample_ids) != cache.shape[0]:
        raise AssertionError("Sequence cache and sample IDs do not align.")

    statistics = {}
    for name in [
        "ask_volume_1",
        "bid_volume_1",
        "ask_volume_2",
        "bid_volume_2",
    ]:
        statistics[name] = json.loads(
            (train_cache_dir / "statistics" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )

    output = np.empty((cache.shape[0], len(FEATURE_COLUMNS)), dtype=np.float32)
    for start in range(0, cache.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, cache.shape[0])
        batch = build_batch(cache[start:end], statistics)
        if list(batch) != FEATURE_COLUMNS:
            raise AssertionError("Unexpected L1-L2 feature order.")
        for column_index, column_name in enumerate(FEATURE_COLUMNS):
            output[start:end, column_index] = batch[column_name]
        if start % 100_000 == 0:
            print(f"processed {arguments.split} samples {start}:{end}", flush=True)

    if np.isinf(output).any():
        raise AssertionError("L1-L2 shape features contain infinity.")
    data = pd.DataFrame(output, columns=FEATURE_COLUMNS)
    data.insert(0, "sample_id", sample_ids)
    output_path = processed_dir / f"{arguments.split}_market_l1_l2_shape_features.feather"
    safe_write(data, output_path)
    print(f"shape = {data.shape}", flush=True)
    print(f"missing values = {int(data[FEATURE_COLUMNS].isna().sum().sum())}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
