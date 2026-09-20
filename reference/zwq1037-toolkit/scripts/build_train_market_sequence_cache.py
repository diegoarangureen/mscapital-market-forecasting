"""Build a disk-backed market sequence cache with bounded peak memory.

默认只构建前1000个样本的试跑缓存；必须显式传入--full才构建完整缓存。
The default builds a 1,000-sample smoke cache; --full is required for the full cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


MAX_STEPS = 200
BLOCK_SAMPLE_COUNT = 10_000
FULL_TRAIN_SAMPLE_COUNT = 1_064_163
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


def write_json(path: Path, data: dict) -> None:
    """Write a small UTF-8 metadata file."""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_layout(project_dir: Path, smoke_test: bool) -> tuple[np.ndarray, int]:
    """Load compact per-sample counts and validate the sample-ID/month layout."""

    feature_path = project_dir / "data" / "processed" / "train_market_features.feather"
    label_path = project_dir / "data" / "raw" / "label.feather"

    # 两张小表各有一行对应一个样本，不打开2.217亿行宽表。
    # These compact tables have one row per sample; the wide 221.7M-row table stays closed.
    count_data = pd.read_feather(
        feature_path,
        columns=["sample_id", "market_row_count"],
    )
    label_data = pd.read_feather(label_path, columns=["sample_id", "month"])

    expected_ids = np.arange(len(label_data), dtype=np.int64)
    assert np.array_equal(label_data["sample_id"].to_numpy(), expected_ids)
    assert np.array_equal(count_data["sample_id"].to_numpy(), expected_ids)
    assert label_data["month"].is_monotonic_increasing
    assert int((label_data["month"] <= 59).sum()) == FULL_TRAIN_SAMPLE_COUNT

    if smoke_test:
        sample_count = 1_000
    else:
        sample_count = len(label_data)

    counts = count_data["market_row_count"].to_numpy(dtype=np.int32)[:sample_count]
    assert len(counts) == sample_count
    assert (counts > 0).all() and (counts <= 212).all()
    return counts, sample_count


def make_row_starts(counts: np.ndarray) -> np.ndarray:
    """Return the first raw-row offset of every sample plus the final end offset."""

    starts = np.empty(len(counts) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, dtype=np.int64, out=starts[1:])
    return starts


def audit_raw_sample_order(
    market_path: Path,
    counts_path: Path,
) -> None:
    """Verify that raw rows form consecutive sample-ID groups matching saved counts."""

    counts = np.load(counts_path, mmap_mode="r")
    starts = make_row_starts(counts)
    raw_row_count = int(starts[-1])

    # 只读sample_id一列；子进程退出后整列内存会被操作系统释放。
    # Read only sample_id; the child process releases the full column on exit.
    sample_ids = pl.read_ipc(
        market_path,
        columns=["sample_id"],
        n_rows=raw_row_count,
        memory_map=False,
    )["sample_id"].to_numpy()
    assert len(sample_ids) == raw_row_count

    for sample_start in range(0, len(counts), BLOCK_SAMPLE_COUNT):
        sample_end = min(sample_start + BLOCK_SAMPLE_COUNT, len(counts))
        raw_start = int(starts[sample_start])
        raw_end = int(starts[sample_end])
        expected = np.repeat(
            np.arange(sample_start, sample_end, dtype=np.int32),
            counts[sample_start:sample_end],
        )
        if not np.array_equal(sample_ids[raw_start:raw_end], expected):
            raise RuntimeError(
                f"Raw sample order mismatch in sample range {sample_start}:{sample_end}."
            )

    print(f"layout audit passed: samples={len(counts)}, raw rows={raw_row_count}")


def block_positions(
    counts: np.ndarray,
    starts: np.ndarray,
    sample_start: int,
    sample_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build source-row indices and a valid-position mask for one sample block."""

    block_counts = counts[sample_start:sample_end].astype(np.int64, copy=False)
    kept_counts = np.minimum(block_counts, MAX_STEPS)
    destination_starts = MAX_STEPS - kept_counts
    positions = np.arange(MAX_STEPS, dtype=np.int64)[None, :]
    valid_positions = positions >= destination_starts[:, None]

    kept_raw_starts = (
        starts[sample_start:sample_end]
        + block_counts
        - kept_counts
    )
    source_indices = (
        kept_raw_starts[:, None]
        + positions
        - destination_starts[:, None]
    )
    return source_indices, valid_positions


def compute_training_statistics(
    values: np.ndarray,
    starts: np.ndarray,
    training_sample_count: int,
) -> tuple[float, float]:
    """Compute mean/std from real rows of training months only."""

    training_raw_end = int(starts[training_sample_count])
    training_values = values[:training_raw_end]
    mean_value = float(np.nanmean(training_values, dtype=np.float64))
    scale_value = float(np.nanstd(training_values, dtype=np.float64))
    if not np.isfinite(mean_value):
        raise ValueError("Training mean is not finite.")
    if not np.isfinite(scale_value) or scale_value == 0.0:
        scale_value = 1.0
    return mean_value, scale_value


def write_one_source_column(
    market_path: Path,
    cache_path: Path,
    counts_path: Path,
    source_column: str,
    channel_index: int,
    training_sample_count: int,
    statistics_path: Path,
) -> None:
    """Read one full source column, transform it, and write one cache channel."""

    counts = np.load(counts_path, mmap_mode="r")
    starts = make_row_starts(counts)
    raw_row_count = int(starts[-1])

    # 每个子进程只解压一个原始列，避免13列宽表同时进入内存。
    # Each child decompresses one source column, never the full 13-column table.
    values = pl.read_ipc(
        market_path,
        columns=[source_column],
        n_rows=raw_row_count,
        memory_map=False,
    )[source_column].to_numpy()
    assert len(values) == raw_row_count

    mean_value = None
    scale_value = None
    if source_column in VALUE_COLUMNS:
        mean_value, scale_value = compute_training_statistics(
            values,
            starts,
            training_sample_count,
        )

    cache = np.lib.format.open_memmap(cache_path, mode="r+")
    for sample_start in range(0, len(counts), BLOCK_SAMPLE_COUNT):
        sample_end = min(sample_start + BLOCK_SAMPLE_COUNT, len(counts))
        source_indices, valid_positions = block_positions(
            counts,
            starts,
            sample_start,
            sample_end,
        )

        # 每个块先保持全0；只在真实观测位置写入，padding不会参与标准化。
        # Start each block at zero and transform observed positions only, never padding.
        output_block = np.zeros(
            (sample_end - sample_start, MAX_STEPS),
            dtype=np.float32,
        )
        selected_values = values[source_indices[valid_positions]].astype(
            np.float32,
            copy=False,
        )

        if source_column == "seconds_before_predict":
            selected_values = selected_values / 600.0
        else:
            selected_values = (selected_values - mean_value) / scale_value
            selected_values = np.nan_to_num(
                selected_values,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        output_block[valid_positions] = selected_values
        cache[sample_start:sample_end, channel_index, :] = output_block

        # transaction_count同一次读取顺便生成有无成交通道，避免重复解压整列。
        # Reuse transaction_count to write the activity channel without rereading it.
        if source_column == "transaction_count":
            activity_block = np.zeros_like(output_block)
            activity_block[valid_positions] = (
                values[source_indices[valid_positions]] > 0
            ).astype(np.float32)
            cache[
                sample_start:sample_end,
                HAS_TRANSACTIONS_CHANNEL_INDEX,
                :,
            ] = activity_block

    cache.flush()
    del cache

    statistics = {
        "source_column": source_column,
        "channel_index": channel_index,
        "mean": mean_value,
        "scale": scale_value,
        "training_sample_count": training_sample_count,
    }
    write_json(statistics_path, statistics)
    print(f"finished source column={source_column}, cache channel={channel_index}")


def validate_countdown_order(
    market_path: Path,
    counts_path: Path,
) -> None:
    """Verify descending countdown within every sample using bounded blocks."""

    counts = np.load(counts_path, mmap_mode="r")
    starts = make_row_starts(counts)
    raw_row_count = int(starts[-1])
    seconds = pl.read_ipc(
        market_path,
        columns=["seconds_before_predict"],
        n_rows=raw_row_count,
        memory_map=False,
    )["seconds_before_predict"].to_numpy()

    for sample_start in range(0, len(counts), BLOCK_SAMPLE_COUNT):
        sample_end = min(sample_start + BLOCK_SAMPLE_COUNT, len(counts))
        raw_start = int(starts[sample_start])
        raw_end = int(starts[sample_end])
        differences = np.diff(seconds[raw_start:raw_end])
        boundary_indices = (
            np.cumsum(counts[sample_start:sample_end], dtype=np.int64)[:-1] - 1
        )
        differences[boundary_indices] = 0.0
        if not (differences <= 0.0).all():
            raise RuntimeError(
                f"Countdown order mismatch in sample range {sample_start}:{sample_end}."
            )
    print("countdown order audit passed")


def write_row_mask(cache_path: Path, counts_path: Path) -> None:
    """Write 1 for observed positions and 0 for left padding."""

    counts = np.load(counts_path, mmap_mode="r")
    starts = make_row_starts(counts)
    cache = np.lib.format.open_memmap(cache_path, mode="r+")
    for sample_start in range(0, len(counts), BLOCK_SAMPLE_COUNT):
        sample_end = min(sample_start + BLOCK_SAMPLE_COUNT, len(counts))
        _, valid_positions = block_positions(
            counts,
            starts,
            sample_start,
            sample_end,
        )
        cache[sample_start:sample_end, ROW_MASK_CHANNEL_INDEX, :] = (
            valid_positions.astype(CACHE_DTYPE)
        )
    cache.flush()
    del cache
    print("finished row_mask channel")


def cache_paths(project_dir: Path, smoke_test: bool) -> dict[str, Path]:
    """Return isolated paths so smoke output can never overwrite full output."""

    if smoke_test:
        cache_dir = project_dir / "data" / "interim" / "sequence_cache_smoke_v1"
    else:
        cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    return {
        "dir": cache_dir,
        "cache": cache_dir / "sequences.npy",
        "counts": cache_dir / "sample_counts.npy",
        "manifest": cache_dir / "manifest.json",
        "statistics_dir": cache_dir / "statistics",
    }


def make_manifest(sample_count: int, smoke_test: bool) -> dict:
    """Return the fixed configuration and empty resumability state."""

    return {
        "version": 1,
        "status": "building",
        "smoke_test": smoke_test,
        "sample_count": sample_count,
        "shape": [sample_count, NUM_CHANNELS, MAX_STEPS],
        "dtype": str(np.dtype(CACHE_DTYPE)),
        "train_months_for_statistics": [0, 59],
        "layout_audited": False,
        "completed_channels": [],
    }


def load_or_create_manifest(
    paths: dict[str, Path],
    sample_count: int,
    smoke_test: bool,
) -> dict:
    """Create new cache metadata or safely resume an identical configuration."""

    expected = make_manifest(sample_count, smoke_test)
    cache_path = paths["cache"]
    manifest_path = paths["manifest"]

    if manifest_path.exists():
        if not cache_path.exists():
            raise RuntimeError("Manifest exists but sequence cache is missing.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ["version", "smoke_test", "sample_count", "shape", "dtype"]:
            if manifest.get(key) != expected[key]:
                raise RuntimeError(f"Existing cache configuration mismatch: {key}")
        print("resuming existing cache; completed =", manifest["completed_channels"])
        return manifest

    if cache_path.exists():
        raise RuntimeError(
            "Cache exists without a manifest; refusing to overwrite it automatically."
        )

    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=tuple(expected["shape"]),
    )
    cache.flush()
    del cache
    write_json(manifest_path, expected)
    return expected


def run_child(command_arguments: list[str]) -> None:
    """Run one memory-heavy operation in an isolated Python process."""

    command = [sys.executable, str(Path(__file__).resolve())] + command_arguments
    subprocess.run(command, check=True)


def build_cache(project_dir: Path, smoke_test: bool) -> None:
    """Coordinate audited, resumable, one-column-at-a-time cache construction."""

    market_path = project_dir / "data" / "raw" / "train" / "market.feather"
    paths = cache_paths(project_dir, smoke_test)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["statistics_dir"].mkdir(parents=True, exist_ok=True)

    counts, sample_count = load_layout(project_dir, smoke_test)
    np.save(paths["counts"], counts)
    training_sample_count = min(FULL_TRAIN_SAMPLE_COUNT, sample_count)

    # 新建缓存前检查磁盘余量；除缓存本体外保留2GiB安全空间。
    # Check free disk space before creation and keep a 2-GiB safety margin.
    if not paths["cache"].exists():
        required_bytes = (
            sample_count
            * NUM_CHANNELS
            * MAX_STEPS
            * np.dtype(CACHE_DTYPE).itemsize
        )
        safety_margin_bytes = 2 * 1024 ** 3
        free_bytes = shutil.disk_usage(paths["dir"]).free
        print(f"required cache GiB = {required_bytes / (1024 ** 3):.2f}")
        print(f"free disk GiB = {free_bytes / (1024 ** 3):.2f}")
        if free_bytes < required_bytes + safety_margin_bytes:
            raise OSError("Not enough free disk space for cache plus safety margin.")

    # 先检查原始排列再创建/续写正式通道；不把注释中的假设当作证据。
    # Audit raw ordering before channel work instead of trusting a comment as evidence.
    manifest = load_or_create_manifest(paths, sample_count, smoke_test)
    if not manifest["layout_audited"]:
        run_child([
            "--audit-layout",
            "--market-path", str(market_path),
            "--counts-path", str(paths["counts"]),
        ])
        manifest["layout_audited"] = True
        write_json(paths["manifest"], manifest)

    if "row_mask" not in manifest["completed_channels"]:
        write_row_mask(paths["cache"], paths["counts"])
        manifest["completed_channels"].append("row_mask")
        write_json(paths["manifest"], manifest)

    for channel_index, source_column in enumerate(VALUE_COLUMNS):
        if source_column in manifest["completed_channels"]:
            print(f"reusing completed source column={source_column}")
            continue
        statistics_path = paths["statistics_dir"] / f"{source_column}.json"
        run_child([
            "--write-column",
            "--market-path", str(market_path),
            "--cache-path", str(paths["cache"]),
            "--counts-path", str(paths["counts"]),
            "--source-column", source_column,
            "--channel-index", str(channel_index),
            "--training-sample-count", str(training_sample_count),
            "--statistics-path", str(statistics_path),
        ])
        manifest["completed_channels"].append(source_column)
        if source_column == "transaction_count":
            manifest["completed_channels"].append("has_transactions")
        write_json(paths["manifest"], manifest)

    if "seconds_before_predict" not in manifest["completed_channels"]:
        run_child([
            "--validate-countdown",
            "--market-path", str(market_path),
            "--counts-path", str(paths["counts"]),
        ])
        statistics_path = paths["statistics_dir"] / "seconds_before_predict.json"
        run_child([
            "--write-column",
            "--market-path", str(market_path),
            "--cache-path", str(paths["cache"]),
            "--counts-path", str(paths["counts"]),
            "--source-column", "seconds_before_predict",
            "--channel-index", str(SECONDS_CHANNEL_INDEX),
            "--training-sample-count", str(training_sample_count),
            "--statistics-path", str(statistics_path),
        ])
        manifest["completed_channels"].append("seconds_before_predict")

    manifest["status"] = "complete"
    write_json(paths["manifest"], manifest)

    cache = np.load(paths["cache"], mmap_mode="r")
    assert cache.shape == (sample_count, NUM_CHANNELS, MAX_STEPS)
    assert cache.dtype == CACHE_DTYPE
    assert isinstance(cache, np.memmap)
    check_indices = np.unique(
        np.array([0, min(1, sample_count - 1), sample_count - 1], dtype=np.int64)
    )
    assert np.isfinite(cache[check_indices]).all()
    observed_counts = cache[check_indices, ROW_MASK_CHANNEL_INDEX, :].sum(axis=1)

    print("cache status =", manifest["status"])
    print("cache path =", paths["cache"])
    print("cache shape =", cache.shape)
    print("cache dtype =", cache.dtype)
    print("checked observed counts =", observed_counts.astype(int).tolist())
    if smoke_test:
        print("SMOKE ONLY: inspect this result before running with --full.")


def parse_arguments() -> argparse.Namespace:
    """Parse driver and isolated-child modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--audit-layout", action="store_true")
    parser.add_argument("--validate-countdown", action="store_true")
    parser.add_argument("--write-column", action="store_true")
    parser.add_argument("--market-path", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--counts-path", type=Path)
    parser.add_argument("--source-column")
    parser.add_argument("--channel-index", type=int)
    parser.add_argument("--training-sample-count", type=int)
    parser.add_argument("--statistics-path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.audit_layout:
        audit_raw_sample_order(arguments.market_path, arguments.counts_path)
    elif arguments.validate_countdown:
        validate_countdown_order(arguments.market_path, arguments.counts_path)
    elif arguments.write_column:
        write_one_source_column(
            arguments.market_path,
            arguments.cache_path,
            arguments.counts_path,
            arguments.source_column,
            arguments.channel_index,
            arguments.training_sample_count,
            arguments.statistics_path,
        )
    else:
        current_project_dir = Path(__file__).resolve().parents[1]
        build_cache(current_project_dir, smoke_test=not arguments.full)
