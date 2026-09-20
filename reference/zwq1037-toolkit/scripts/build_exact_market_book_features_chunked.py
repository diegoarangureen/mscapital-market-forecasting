"""Build exact book features in restartable, low-memory sample chunks."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

# 限制 Polars 工作线程，降低笔记本电脑的持续 CPU 与内存压力。
# Limit Polars workers to reduce sustained CPU and memory pressure on the laptop.
os.environ.setdefault("POLARS_MAX_THREADS", "2")

import numpy as np
import polars as pl

import build_exact_market_book_features as exact_builder
from build_market_aggregate_features import safe_write


SOURCE_COLUMNS = [
    "sample_id",
    "seconds_before_predict",
    "ask_price_1",
    "ask_volume_1",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "bid_price_2",
    "bid_volume_2",
]


def build_one_chunk(
    project_dir: Path,
    split: str,
    source_chunk: pl.DataFrame,
) -> pl.DataFrame:
    """Reuse the audited feature definitions without scanning or sorting globally."""

    original_scan_ipc = pl.scan_ipc
    original_sort = pl.LazyFrame.sort

    def scan_current_chunk(*args, **kwargs):
        del args, kwargs
        return source_chunk.lazy()

    def skip_verified_source_sort(self, by, *args, **kwargs):
        if by == ["sample_id", "seconds_before_predict"]:
            return self
        return original_sort(self, by, *args, **kwargs)

    pl.scan_ipc = scan_current_chunk
    pl.LazyFrame.sort = skip_verified_source_sort
    try:
        return exact_builder.build_features(project_dir, split)
    finally:
        pl.scan_ipc = original_scan_ipc
        pl.LazyFrame.sort = original_sort


def completed_chunk_is_valid(
    chunk_path: Path,
    expected_sample_ids: np.ndarray,
) -> bool:
    if not chunk_path.exists():
        return False
    try:
        saved_ids = (
            pl.read_ipc(chunk_path, columns=["sample_id"])
            .sort("sample_id")
            .get_column("sample_id")
            .to_numpy()
        )
    except Exception:
        return False
    return np.array_equal(saved_ids, np.sort(expected_sample_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--chunk-samples", type=int, default=1000)
    arguments = parser.parse_args()
    if arguments.chunk_samples <= 0:
        raise ValueError("--chunk-samples must be positive.")

    project_dir = Path(__file__).resolve().parents[1]
    market_path = project_dir / "data" / "raw" / arguments.split / "market.feather"
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / f"exact_book_{arguments.split}_chunks_{arguments.chunk_samples}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"POLARS_MAX_THREADS={pl.thread_pool_size()}", flush=True)
    print(f"loading selected source columns from {market_path}", flush=True)
    source = pl.read_ipc(
        market_path,
        columns=SOURCE_COLUMNS,
        memory_map=True,
        rechunk=False,
    )
    print(f"source shape={source.shape}", flush=True)

    # 原始数据已经审计为按 sample_id 连续排列；这里只计算每组行数以确定安全切片边界。
    # The source was audited as contiguous by sample_id; group sizes define safe slices.
    sample_counts = source.group_by("sample_id", maintain_order=True).len()
    sample_ids = sample_counts.get_column("sample_id").to_numpy()
    row_counts = sample_counts.get_column("len").to_numpy()
    row_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(row_counts, dtype=np.int64)]
    )
    if int(row_offsets[-1]) != source.height:
        raise AssertionError("Sample row counts do not cover the source table.")

    chunk_paths: list[Path] = []
    chunk_total = (
        len(sample_ids) + arguments.chunk_samples - 1
    ) // arguments.chunk_samples
    for chunk_index, sample_start in enumerate(
        range(0, len(sample_ids), arguments.chunk_samples)
    ):
        sample_stop = min(sample_start + arguments.chunk_samples, len(sample_ids))
        expected_ids = sample_ids[sample_start:sample_stop]
        chunk_path = run_dir / f"chunk_{chunk_index:05d}.feather"
        chunk_paths.append(chunk_path)
        if completed_chunk_is_valid(chunk_path, expected_ids):
            print(
                f"chunk {chunk_index + 1}/{chunk_total}: checkpoint exists",
                flush=True,
            )
            continue

        row_start = int(row_offsets[sample_start])
        row_stop = int(row_offsets[sample_stop])
        source_chunk = source.slice(row_start, row_stop - row_start)
        output_chunk = build_one_chunk(project_dir, arguments.split, source_chunk)
        output_chunk = output_chunk.sort("sample_id")
        if not np.array_equal(
            output_chunk.get_column("sample_id").to_numpy(),
            np.sort(expected_ids),
        ):
            raise AssertionError(f"Unexpected sample ids in chunk {chunk_index}.")
        safe_write(output_chunk, chunk_path)
        print(
            f"chunk {chunk_index + 1}/{chunk_total}: "
            f"rows={source_chunk.height}, samples={output_chunk.height}",
            flush=True,
        )
        del source_chunk, output_chunk
        gc.collect()

    missing_chunks = [path for path in chunk_paths if not path.exists()]
    if missing_chunks:
        raise AssertionError(f"Missing {len(missing_chunks)} chunk checkpoints.")
    output = pl.concat(
        [pl.read_ipc(path, rechunk=False) for path in chunk_paths],
        how="vertical",
        rechunk=False,
    ).sort("sample_id")
    if output.height != len(sample_ids):
        raise AssertionError("Final feature table has an unexpected sample count.")

    output_path = (
        project_dir
        / "data"
        / "processed"
        / f"{arguments.split}_market_book_features_exact.feather"
    )
    safe_write(output, output_path)
    print(f"shape={output.shape}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
