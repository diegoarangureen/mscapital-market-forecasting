"""Read-only validation for the completed full train sequence cache."""

import json
from pathlib import Path

import numpy as np


project_dir = Path(__file__).resolve().parents[1]
cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
counts = np.load(cache_dir / "sample_counts.npy", mmap_mode="r")

assert isinstance(cache, np.memmap)
assert cache.shape == (1_257_637, 14, 200)
assert cache.dtype == np.float16

all_finite = True
all_masks_binary = True
all_counts_match = True
all_padding_zero = True

# 每次只验证10000个样本，避免检查程序本身把完整缓存载入内存。
# Validate 10,000 samples at a time so the checker never loads the full cache.
for sample_start in range(0, len(counts), 10_000):
    sample_end = min(sample_start + 10_000, len(counts))
    block = cache[sample_start:sample_end]
    masks = block[:, 13, :]

    all_finite = all_finite and bool(np.isfinite(block).all())
    all_masks_binary = all_masks_binary and bool(
        ((masks == 0) | (masks == 1)).all()
    )
    expected_counts = np.minimum(counts[sample_start:sample_end], 200)
    actual_counts = masks.sum(axis=1).astype(np.int32)
    all_counts_match = all_counts_match and bool(
        np.array_equal(actual_counts, expected_counts)
    )

    padding_positions = masks == 0
    for channel_index in range(13):
        channel_padding = block[:, channel_index, :][padding_positions]
        all_padding_zero = all_padding_zero and bool((channel_padding == 0).all())

    if not (
        all_finite
        and all_masks_binary
        and all_counts_match
        and all_padding_zero
    ):
        raise AssertionError(f"Cache validation failed near sample {sample_start}.")

statistics_paths = sorted((cache_dir / "statistics").glob("*.json"))
assert len(statistics_paths) == 12
for statistics_path in statistics_paths:
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    assert statistics["training_sample_count"] == 1_064_163

print("type / shape / dtype =", type(cache), cache.shape, cache.dtype)
print("all values finite =", all_finite)
print("all masks binary =", all_masks_binary)
print("all observed counts match =", all_counts_match)
print("all padding values zero =", all_padding_zero)
print(
    "cache GiB =",
    round((cache_dir / "sequences.npy").stat().st_size / (1024 ** 3), 4),
)
print("statistics files =", len(statistics_paths))
print("all statistics training sample count = 1064163")
