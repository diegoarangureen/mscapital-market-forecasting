"""Build the compact final-15-second market block from the public LGB baseline.

The implementation reuses the audited 200-step sequence caches, reverses their
training-only scaling, and computes every feature independently inside one
sample_id. It avoids both cross-month aggregation and another raw-table scan.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_train_market_microstructure_features import CHANNELS, unscale


BATCH_SIZE = 10_000
FEATURE_COLUMNS = [
    "m_mid_mean_15",
    "m_mid_std_15",
    "m_sp_mean_15",
    "m_imb_mean_15",
    "m_rv_15",
    "m_ofi_sum_15",
    "m_txv_sum_15",
    "x_rv_15_over_full",
]


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1, dtype=np.float64)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def masked_sample_std(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match Polars' default sample standard deviation (ddof=1)."""

    count = mask.sum(axis=1)
    mean = masked_mean(values, mask)
    squared = np.where(mask, np.square(values - mean[:, None]), 0.0).sum(
        axis=1, dtype=np.float64
    )
    return np.sqrt(
        np.divide(
            squared,
            count - 1,
            out=np.full(len(values), np.nan),
            where=count > 1,
        )
    )


def build_batch(cache_block: np.ndarray, statistics: dict) -> dict[str, np.ndarray]:
    """Reproduce the public baseline's seven 15-second stats and RV ratio."""

    row_mask = cache_block[:, CHANNELS["row_mask"], :] > 0.5
    seconds = (
        cache_block[:, CHANNELS["seconds_before_predict"], :].astype(np.float64)
        * 600.0
    )
    window_15 = row_mask & (seconds <= 15.0)

    ask = unscale(cache_block, "ask_price_1", statistics)
    bid = unscale(cache_block, "bid_price_1", statistics)
    ask_volume = unscale(cache_block, "ask_volume_1", statistics)
    bid_volume = unscale(cache_block, "bid_volume_1", statistics)
    transaction_volume = unscale(cache_block, "transaction_volume", statistics)

    mid = (ask + bid) * 0.5
    spread = ask - bid
    imbalance = np.divide(
        ask_volume - bid_volume,
        ask_volume + bid_volume + 1.0,
        out=np.full_like(ask_volume, np.nan),
        where=row_mask,
    )

    # Differentiate the full chronological sequence first, then filter changes.
    dmid = np.zeros_like(mid)
    ask_change = np.zeros_like(ask_volume)
    bid_change = np.zeros_like(bid_volume)
    pair_mask = row_mask[:, 1:] & row_mask[:, :-1]
    dmid[:, 1:] = np.where(pair_mask, mid[:, 1:] - mid[:, :-1], 0.0)
    ask_change[:, 1:] = np.where(
        pair_mask, ask_volume[:, 1:] - ask_volume[:, :-1], 0.0
    )
    bid_change[:, 1:] = np.where(
        pair_mask, bid_volume[:, 1:] - bid_volume[:, :-1], 0.0
    )
    ofi = ask_change - bid_change

    rv_full = np.sqrt(
        np.where(row_mask, np.square(dmid), 0.0).sum(axis=1, dtype=np.float64)
    )
    rv_15 = np.sqrt(
        np.where(window_15, np.square(dmid), 0.0).sum(axis=1, dtype=np.float64)
    )
    rv_ratio = np.divide(
        rv_15,
        rv_full + 1e-8,
        out=np.full(len(mid), np.nan),
        where=np.isfinite(rv_full),
    )
    return {
        "m_mid_mean_15": masked_mean(mid, window_15),
        "m_mid_std_15": masked_sample_std(mid, window_15),
        "m_sp_mean_15": masked_mean(spread, window_15),
        "m_imb_mean_15": masked_mean(imbalance, window_15),
        "m_rv_15": rv_15,
        "m_ofi_sum_15": np.where(window_15, ofi, 0.0).sum(
            axis=1, dtype=np.float64
        ),
        "m_txv_sum_15": np.where(window_15, transaction_volume, 0.0).sum(
            axis=1, dtype=np.float64
        ),
        "x_rv_15_over_full": rv_ratio,
    }


def safe_write(data: pd.DataFrame, output_path: Path) -> None:
    temporary_file = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        data.to_feather(temporary_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    for name in [
        "ask_price_1",
        "bid_price_1",
        "ask_volume_1",
        "bid_volume_1",
        "transaction_volume",
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
            raise AssertionError("Unexpected 15-second feature order.")
        for column_index, column_name in enumerate(FEATURE_COLUMNS):
            output[start:end, column_index] = batch[column_name]
        if start % 100_000 == 0:
            print(f"processed {arguments.split} samples {start}:{end}", flush=True)

    if np.isinf(output).any():
        raise AssertionError("15-second features contain infinity.")
    feature_data = pd.DataFrame(output, columns=FEATURE_COLUMNS)
    feature_data.insert(0, "sample_id", sample_ids)
    output_path = processed_dir / f"{arguments.split}_market_last15_baseline_features.feather"
    safe_write(feature_data, output_path)
    print(f"shape = {feature_data.shape}", flush=True)
    print(f"missing values = {int(feature_data[FEATURE_COLUMNS].isna().sum().sum())}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
