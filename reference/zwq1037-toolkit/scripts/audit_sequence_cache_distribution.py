"""Audit train/test market-sequence length, truncation, and time coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ID = "EXP-AUDIT-002-SEQUENCE-DOMAIN"
MAX_STEPS = 200
BATCH_SIZE = 20_000
SECONDS_CHANNEL = 11
ACTIVITY_CHANNEL = 12
MASK_CHANNEL = 13


def summarize(name: str, values: np.ndarray) -> list[dict[str, object]]:
    finite = values[np.isfinite(values)].astype(np.float64, copy=False)
    if finite.size == 0:
        return [{"metric": name, "count": 0}]
    quantiles = np.quantile(finite, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return [
        {
            "metric": name,
            "count": int(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "min": float(finite.min()),
            "q01": float(quantiles[0]),
            "q10": float(quantiles[1]),
            "q25": float(quantiles[2]),
            "q50": float(quantiles[3]),
            "q75": float(quantiles[4]),
            "q90": float(quantiles[5]),
            "q99": float(quantiles[6]),
            "max": float(finite.max()),
        }
    ]


def extract_cache_metrics(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    counts = np.load(cache_dir / "sample_counts.npy", mmap_mode="r")
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    if cache.shape != (len(counts), 14, MAX_STEPS):
        raise AssertionError(f"Unexpected cache shape: {cache.shape}")

    kept = np.minimum(np.asarray(counts, dtype=np.int32), MAX_STEPS)
    earliest_seconds = np.empty(len(counts), dtype=np.float32)
    latest_seconds = np.empty(len(counts), dtype=np.float32)
    coverage_seconds = np.empty(len(counts), dtype=np.float32)
    activity_ratio = np.empty(len(counts), dtype=np.float32)
    mask_mismatch_count = 0
    countdown_violation_count = 0

    for start in range(0, len(counts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(counts))
        block_size = end - start
        block_kept = kept[start:end]
        first_positions = MAX_STEPS - block_kept
        row_indices = np.arange(block_size)
        seconds = np.asarray(cache[start:end, SECONDS_CHANNEL, :], dtype=np.float32)
        activity = np.asarray(cache[start:end, ACTIVITY_CHANNEL, :], dtype=np.float32)
        mask = np.asarray(cache[start:end, MASK_CHANNEL, :], dtype=np.float32) > 0.5

        earliest = seconds[row_indices, first_positions] * 600.0
        latest = seconds[:, -1] * 600.0
        earliest_seconds[start:end] = earliest
        latest_seconds[start:end] = latest
        coverage_seconds[start:end] = earliest - latest
        activity_ratio[start:end] = np.divide(
            (activity * mask).sum(axis=1),
            block_kept,
            out=np.zeros(block_size, dtype=np.float32),
            where=block_kept > 0,
        )
        mask_mismatch_count += int(np.count_nonzero(mask.sum(axis=1) != block_kept))
        sampled_rows = np.arange(0, block_size, 20)
        if sampled_rows.size:
            sampled_seconds = seconds[sampled_rows]
            sampled_mask = mask[sampled_rows]
            differences = np.diff(sampled_seconds, axis=1)
            valid_pairs = sampled_mask[:, :-1] & sampled_mask[:, 1:]
            countdown_violation_count += int(
                np.count_nonzero(np.any((differences > 1.0e-4) & valid_pairs, axis=1))
            )
        if start % 200_000 == 0:
            print(f"audited {cache_dir.name} rows {start}:{end}", flush=True)

    checks = {
        "cache_shape": list(cache.shape),
        "count_min": int(np.min(counts)),
        "count_max": int(np.max(counts)),
        "empty_sequence_count": int(np.count_nonzero(counts <= 0)),
        "mask_mismatch_count": mask_mismatch_count,
        "sampled_countdown_violation_count": countdown_violation_count,
    }
    return {
        "raw_count": np.asarray(counts, dtype=np.float32),
        "kept_count": kept.astype(np.float32),
        "earliest_seconds": earliest_seconds,
        "latest_seconds": latest_seconds,
        "coverage_seconds": coverage_seconds,
        "activity_ratio": activity_ratio,
    }, checks


def group_rows(
    split_name: str,
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected_count = metrics["raw_count"][mask]
    base = {
        "split": split_name,
        "sample_count": int(mask.sum()),
        "cap200_rate": float(np.mean(selected_count >= MAX_STEPS)),
        "truncated_over200_rate": float(np.mean(selected_count > MAX_STEPS)),
    }
    for metric_name, values in metrics.items():
        for row in summarize(metric_name, values[mask]):
            rows.append({**base, **row})
    return rows


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month"],
    ).sort_values("sample_id").reset_index(drop=True)
    template = pd.read_csv(
        project_dir / "data" / "raw" / "submission.csv", usecols=["sample_id"]
    )
    if not np.array_equal(labels["sample_id"].to_numpy(), np.arange(len(labels))):
        raise AssertionError("Train labels are not aligned to cache positions.")
    if not np.array_equal(template["sample_id"].to_numpy(), np.arange(len(template))):
        raise AssertionError("Test template is not aligned to cache positions.")

    train_metrics, train_checks = extract_cache_metrics(
        project_dir / "data" / "processed" / "train_sequence_cache_v1"
    )
    test_metrics, test_checks = extract_cache_metrics(
        project_dir / "data" / "processed" / "test_sequence_cache_v1"
    )
    months = labels["month"].to_numpy()
    rows: list[dict[str, object]] = []
    for name, start, end in (
        ("train_0_39", 0, 39),
        ("train_40_49", 40, 49),
        ("train_50_59", 50, 59),
        ("train_60_70", 60, 70),
        ("train_0_70", 0, 70),
    ):
        rows.extend(group_rows(name, train_metrics, (months >= start) & (months <= end)))
    rows.extend(
        group_rows("test", test_metrics, np.ones(len(template), dtype=bool))
    )
    distribution = pd.DataFrame(rows)
    distribution.to_csv(run_dir / "sequence_distribution.csv", index=False)

    key_metrics = ["raw_count", "coverage_seconds", "activity_ratio"]
    comparison = []
    for metric in key_metrics:
        train_values = train_metrics[metric]
        test_values = test_metrics[metric]
        comparison.append(
            {
                "metric": metric,
                "train_0_70_mean": float(np.mean(train_values)),
                "train_60_70_mean": float(np.mean(train_values[months >= 60])),
                "test_mean": float(np.mean(test_values)),
                "test_over_train_mean_ratio": float(
                    np.mean(test_values) / np.mean(train_values)
                ),
                "test_minus_train_60_70_mean": float(
                    np.mean(test_values) - np.mean(train_values[months >= 60])
                ),
            }
        )
    pd.DataFrame(comparison).to_csv(run_dir / "key_comparison.csv", index=False)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "audit market sequence comparability before GRU experiments",
        "train_checks": train_checks,
        "test_checks": test_checks,
        "key_comparison": comparison,
        "sequence_distribution_path": str(run_dir / "sequence_distribution.csv"),
    }
    (run_dir / "sequence_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
