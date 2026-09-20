"""Build the test market sequence cache with train-fitted scaling statistics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl


MAX_STEPS = 200
BLOCK_SAMPLE_COUNT = 10_000
CACHE_DTYPE = np.float16
VALUE_COLUMNS = [
    "transaction_avgprice",
    "transaction_volume",
    "transaction_count",
    "ask_price_1",
    "ask_volume_1",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "bid_price_2",
    "bid_volume_2",
]
SECONDS_CHANNEL_INDEX = len(VALUE_COLUMNS)
HAS_TRANSACTIONS_CHANNEL_INDEX = SECONDS_CHANNEL_INDEX + 1
ROW_MASK_CHANNEL_INDEX = HAS_TRANSACTIONS_CHANNEL_INDEX + 1
NUM_CHANNELS = ROW_MASK_CHANNEL_INDEX + 1


def make_starts(counts: np.ndarray) -> np.ndarray:
    """Return the first raw-row offset for every sample plus the final end."""

    starts = np.empty(len(counts) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, dtype=np.int64, out=starts[1:])
    return starts


def block_positions(counts, starts, sample_start, sample_end):
    """Return source-row positions and valid destinations for one sample block."""

    block_counts = counts[sample_start:sample_end].astype(np.int64, copy=False)
    kept_counts = np.minimum(block_counts, MAX_STEPS)
    destination_starts = MAX_STEPS - kept_counts
    positions = np.arange(MAX_STEPS, dtype=np.int64)[None, :]
    valid = positions >= destination_starts[:, None]
    kept_raw_starts = starts[sample_start:sample_end] + block_counts - kept_counts
    source_indices = kept_raw_starts[:, None] + positions - destination_starts[:, None]
    return source_indices, valid


def load_train_statistics(train_cache_dir: Path) -> dict[str, dict]:
    """Load scaling learned exclusively from train months 0--59."""

    statistics = {}
    for name in VALUE_COLUMNS:
        statistics[name] = json.loads(
            (train_cache_dir / "statistics" / f"{name}.json").read_text(encoding="utf-8")
        )
    return statistics


def main() -> None:
    """Build all test channels directly; this script intentionally has no smoke mode."""

    project_dir = Path(__file__).resolve().parents[1]
    market_path = project_dir / "data" / "raw" / "test" / "market.feather"
    feature_path = project_dir / "data" / "processed" / "test_market_features.feather"
    submission_path = project_dir / "data" / "raw" / "submission.csv"
    train_cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    cache_dir = project_dir / "data" / "processed" / "test_sequence_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sequences.npy"
    counts_path = cache_dir / "sample_counts.npy"
    manifest_path = cache_dir / "manifest.json"

    counts_frame = pl.read_ipc(feature_path, columns=["sample_id", "market_row_count"])
    expected_ids = pl.read_csv(submission_path, columns=["sample_id"])["sample_id"].to_numpy()
    if not np.array_equal(counts_frame["sample_id"].to_numpy(), expected_ids):
        raise AssertionError("Test feature IDs do not match submission order.")
    counts = counts_frame["market_row_count"].to_numpy().astype(np.int32)
    if not ((counts > 0) & (counts <= 212)).all():
        raise AssertionError("Unexpected test market row counts.")
    np.save(counts_path, counts)
    starts = make_starts(counts)
    raw_row_count = int(starts[-1])
    required_bytes = len(counts) * NUM_CHANNELS * MAX_STEPS * np.dtype(CACHE_DTYPE).itemsize
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < required_bytes + 2 * 1024**3:
        raise OSError("Not enough free disk space for test cache plus 2-GiB margin.")
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=(len(counts), NUM_CHANNELS, MAX_STEPS),
    )
    cache[:] = 0
    cache.flush()
    del cache
    statistics = load_train_statistics(train_cache_dir)

    for channel_index, source_name in enumerate(VALUE_COLUMNS + ["seconds_before_predict"]):
        values = pl.read_ipc(
            market_path,
            columns=[source_name],
            n_rows=raw_row_count,
            memory_map=False,
        )[source_name].to_numpy()
        cache = np.lib.format.open_memmap(cache_path, mode="r+")
        for sample_start in range(0, len(counts), BLOCK_SAMPLE_COUNT):
            sample_end = min(sample_start + BLOCK_SAMPLE_COUNT, len(counts))
            source_indices, valid = block_positions(counts, starts, sample_start, sample_end)
            output = np.zeros((sample_end - sample_start, MAX_STEPS), dtype=np.float32)
            selected = values[source_indices[valid]].astype(np.float32, copy=False)
            if source_name == "seconds_before_predict":
                selected = selected / 600.0
            else:
                selected = (selected - statistics[source_name]["mean"]) / statistics[source_name]["scale"]
                selected = np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
            output[valid] = selected
            cache[sample_start:sample_end, channel_index, :] = output
            if source_name == "transaction_count":
                activity = np.zeros_like(output)
                activity[valid] = (values[source_indices[valid]] > 0).astype(np.float32)
                cache[sample_start:sample_end, HAS_TRANSACTIONS_CHANNEL_INDEX, :] = activity
        cache.flush()
        del cache
        print(f"finished test channel {source_name}", flush=True)

    cache = np.lib.format.open_memmap(cache_path, mode="r+")
    for sample_start in range(0, len(counts), BLOCK_SAMPLE_COUNT):
        sample_end = min(sample_start + BLOCK_SAMPLE_COUNT, len(counts))
        _, valid = block_positions(counts, starts, sample_start, sample_end)
        cache[sample_start:sample_end, ROW_MASK_CHANNEL_INDEX, :] = valid.astype(CACHE_DTYPE)
    cache.flush()
    del cache
    manifest = {
        "version": 1,
        "status": "complete",
        "split": "test",
        "sample_count": len(counts),
        "shape": [len(counts), NUM_CHANNELS, MAX_STEPS],
        "dtype": str(np.dtype(CACHE_DTYPE)),
        "scaling_source": "train months 0-59 statistics from train_sequence_cache_v1",
        "completed_channels": VALUE_COLUMNS + [
            "seconds_before_predict",
            "has_transactions",
            "row_mask",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test cache complete: {manifest['shape']}")


if __name__ == "__main__":
    main()
