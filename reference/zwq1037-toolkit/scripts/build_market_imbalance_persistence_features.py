"""Build three time-weighted final-60-second order-imbalance path features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_market_last15_baseline_features import safe_write
from build_train_market_microstructure_features import CHANNELS, unscale


BATCH_SIZE = 10_000
DEAD_ZONE = 0.05
WINDOW_SECONDS = 60.0
FEATURE_COLUMNS = [
    "imb_signed_occupancy_60",
    "imb_flip_rate_60",
    "imb_terminal_run_signed_60",
]


def build_batch(cache_block: np.ndarray, statistics: dict) -> dict[str, np.ndarray]:
    """Compute persistence with real elapsed time rather than snapshot counts."""

    row_mask = cache_block[:, CHANNELS["row_mask"], :] > 0.5
    seconds = (
        cache_block[:, CHANNELS["seconds_before_predict"], :].astype(np.float64)
        * 600.0
    )
    bid_volume = np.maximum(unscale(cache_block, "bid_volume_1", statistics), 0.0)
    ask_volume = np.maximum(unscale(cache_block, "ask_volume_1", statistics), 0.0)
    imbalance = np.divide(
        bid_volume - ask_volume,
        bid_volume + ask_volume + 1.0,
        out=np.zeros_like(bid_volume),
        where=row_mask,
    )
    state = np.zeros(imbalance.shape, dtype=np.int8)
    state[imbalance > DEAD_ZONE] = 1
    state[imbalance < -DEAD_ZONE] = -1
    state[~row_mask] = 0

    # A snapshot's state is held until the next known snapshot. Intersect every
    # piecewise-constant interval with the final [0, 60] seconds.
    next_seconds = np.zeros_like(seconds)
    next_seconds[:, :-1] = seconds[:, 1:]
    interval_seconds = np.clip(
        np.minimum(seconds, WINDOW_SECONDS) - np.maximum(next_seconds, 0.0),
        0.0,
        None,
    )
    interval_seconds = np.where(row_mask, interval_seconds, 0.0)
    coverage = interval_seconds.sum(axis=1, dtype=np.float64)

    signed_time = (interval_seconds * state).sum(axis=1, dtype=np.float64)
    signed_occupancy = np.divide(
        signed_time,
        coverage,
        out=np.full(len(state), np.nan),
        where=coverage > 0,
    )

    # Neutral observations do not count as flips. A later non-neutral state is
    # compared with the most recent earlier non-neutral state.
    positions = np.arange(state.shape[1], dtype=np.int16)[None, :]
    significant = row_mask & (state != 0)
    last_significant_index = np.maximum.accumulate(
        np.where(significant, positions, -1), axis=1
    )
    safe_index = np.maximum(last_significant_index, 0)
    carried_state = np.take_along_axis(state, safe_index, axis=1)
    carried_state[last_significant_index < 0] = 0
    previous_carried = np.zeros_like(carried_state)
    previous_carried[:, 1:] = carried_state[:, :-1]
    flip_at_snapshot = (
        significant
        & (seconds <= WINDOW_SECONDS)
        & (previous_carried != 0)
        & (state != previous_carried)
    )
    flip_count = flip_at_snapshot.sum(axis=1, dtype=np.float64)
    flip_rate = np.divide(
        flip_count,
        coverage,
        out=np.full(len(state), np.nan),
        where=coverage > 0,
    )

    # Neutral states break the run. Normalize the terminal signed duration by
    # actual observed coverage so short samples stay comparable.
    reverse_valid_offset = np.argmax(row_mask[:, ::-1], axis=1)
    last_valid_index = state.shape[1] - 1 - reverse_valid_offset
    rows = np.arange(len(state))
    has_rows = row_mask.any(axis=1)
    final_state = state[rows, last_valid_index]
    final_state[~has_rows] = 0
    before_or_at_end = positions <= last_valid_index[:, None]
    breaks_terminal_run = row_mask & before_or_at_end & (
        state != final_state[:, None]
    )
    last_break_index = np.max(
        np.where(breaks_terminal_run, positions, -1), axis=1
    )
    terminal_mask = (
        row_mask
        & before_or_at_end
        & (positions > last_break_index[:, None])
        & (state == final_state[:, None])
        & (final_state[:, None] != 0)
    )
    terminal_seconds = np.where(terminal_mask, interval_seconds, 0.0).sum(
        axis=1, dtype=np.float64
    )
    terminal_run_signed = np.divide(
        terminal_seconds * final_state,
        coverage,
        out=np.full(len(state), np.nan),
        where=coverage > 0,
    )
    return {
        "imb_signed_occupancy_60": signed_occupancy,
        "imb_flip_rate_60": flip_rate,
        "imb_terminal_run_signed_60": terminal_run_signed,
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
    if len(sample_ids) != cache.shape[0] or len(np.unique(sample_ids)) != len(sample_ids):
        raise AssertionError("Sequence cache and aggregate sample IDs do not align.")

    statistics = {}
    for name in ["bid_volume_1", "ask_volume_1"]:
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
            raise AssertionError("Unexpected persistence feature order.")
        for column_index, column_name in enumerate(FEATURE_COLUMNS):
            output[start:end, column_index] = batch[column_name]
        if start % 100_000 == 0:
            print(f"processed {arguments.split} samples {start}:{end}", flush=True)

    if np.isinf(output).any():
        raise AssertionError("Persistence features contain infinity.")
    feature_data = pd.DataFrame(output, columns=FEATURE_COLUMNS)
    feature_data.insert(0, "sample_id", sample_ids)
    output_path = (
        processed_dir / f"{arguments.split}_market_imbalance_persistence_features.feather"
    )
    safe_write(feature_data, output_path)
    print(f"shape = {feature_data.shape}", flush=True)
    print(f"missing values = {int(feature_data[FEATURE_COLUMNS].isna().sum().sum())}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
