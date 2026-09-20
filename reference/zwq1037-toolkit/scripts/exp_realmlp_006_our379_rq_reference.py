"""Fair feature replacement: our 379 features in Yunsu's public RQ RealMLP.

The model and training loop come directly from EXP-REALMLP-005.  Only the
feature matrix changes.  The late fold is the first gate against Yunsu's
public-feature reference score of 0.1387444884.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import exp_realmlp_005_yunsu_public_reference as reference

STATIC_CODE = PROJECT / "data/interim/kaggle_kernels/relative319_xs_tabm_notebook"
sys.path.insert(0, str(STATIC_CODE))
import run_extracted as relative_recipe
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_FEATURE_COLUMNS


SOURCE_CACHE = PROJECT / "data/interim/kaggle_relative319_dev"
OUR_CACHE = PROJECT / "data/interim/our379_reference_cache"
ORDER_PATH = PROJECT / "data/processed/train_order_quote_position_features.feather"
RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-006-OUR379-RQ-REFERENCE"
YUNSU_LATE_REFERENCE = 0.13874448835209802
YUNSU_FIRST_REFERENCE = 0.14071099769027952


def ensure_our379_cache() -> tuple[np.memmap, np.memmap, list[str]]:
    """Build one reusable, ID-aligned float32 matrix without a large RAM spike."""
    OUR_CACHE.mkdir(parents=True, exist_ok=True)
    feature_path = OUR_CACHE / "features.npy"
    target_path = OUR_CACHE / "targets.npy"
    names_path = OUR_CACHE / "feature_columns.json"
    ready_path = OUR_CACHE / "READY.json"
    if ready_path.exists() and feature_path.exists() and target_path.exists() and names_path.exists():
        names = json.loads(names_path.read_text(encoding="utf-8"))
        features = np.load(feature_path, mmap_mode="r")
        targets = np.load(target_path, mmap_mode="r")
        if features.shape == (reference.N_ROWS, 379) and len(names) == 379:
            print(f"OUR379 cache ready shape={features.shape}", flush=True)
            return features, targets, names

    base = np.load(SOURCE_CACHE / "features.npy", mmap_mode="r")
    sample_ids = np.load(SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(SOURCE_CACHE / "months.npy", mmap_mode="r")
    targets_source = np.load(SOURCE_CACHE / "targets.npy", mmap_mode="r")
    base_names = json.loads((SOURCE_CACHE / "feature_columns.json").read_text(encoding="utf-8"))
    if base.shape != (reference.N_ROWS, 319):
        raise AssertionError(f"Unexpected Relative319 shape: {base.shape}")

    # 这 40 个跨表相对特征来自已验证的 379 分支。
    # These 40 cross-table relative features are the validated 379 branch additions.
    relative40, relative_names = relative_recipe.add_relative_features(base, months, base_names)
    if relative40.shape != (reference.N_ROWS, 40):
        raise AssertionError(f"Unexpected relative40 shape: {relative40.shape}")

    order_frame = pd.read_feather(ORDER_PATH).sort_values("sample_id")
    if not np.array_equal(order_frame["sample_id"].to_numpy(), sample_ids):
        raise AssertionError("Order20 sample IDs differ from Relative319 IDs")
    order20 = order_frame[ORDER_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    del order_frame
    feature_names = [*base_names, *relative_names, *ORDER_FEATURE_COLUMNS]
    if len(feature_names) != 379:
        raise AssertionError(f"Expected 379 names, got {len(feature_names)}")

    output = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(reference.N_ROWS, 379),
    )
    block_rows = 32768
    for start in range(0, reference.N_ROWS, block_rows):
        stop = min(start + block_rows, reference.N_ROWS)
        output[start:stop, :319] = base[start:stop]
        output[start:stop, 319:359] = relative40[start:stop]
        output[start:stop, 359:] = order20[start:stop]
        if start // (block_rows * 10) != stop // (block_rows * 10):
            print(f"OUR379 cache rows={stop}/{reference.N_ROWS}", flush=True)
    output.flush()
    targets = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(reference.N_ROWS,),
    )
    targets[:] = targets_source
    targets.flush()
    names_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    ready_path.write_text(
        json.dumps({"rows": reference.N_ROWS, "features": 379}, indent=2), encoding="utf-8"
    )
    print("OUR379 cache completed", flush=True)
    return np.load(feature_path, mmap_mode="r"), np.load(target_path, mmap_mode="r"), feature_names


def chunked_absolute_correlations(
    matrix: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Pearson correlations in chunks to keep 379-feature RAM bounded."""
    rows, columns = matrix.shape
    sum_x = np.zeros(columns, dtype=np.float64)
    sum_x2 = np.zeros(columns, dtype=np.float64)
    sum_xy = np.zeros(columns, dtype=np.float64)
    cross = np.zeros((columns, columns), dtype=np.float64)
    sum_y = 0.0
    sum_y2 = 0.0
    block_rows = 8192
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        x = np.asarray(matrix[start:stop], dtype=np.float64)
        y = np.asarray(target[start:stop], dtype=np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += np.einsum("ij,ij->j", x, x)
        sum_xy += x.T @ y
        cross += x.T @ x
        sum_y += float(y.sum())
        sum_y2 += float(np.dot(y, y))
    covariance_xy = sum_xy - sum_x * sum_y / rows
    variance_x = np.maximum(sum_x2 - sum_x * sum_x / rows, 0.0)
    variance_y = max(sum_y2 - sum_y * sum_y / rows, 0.0)
    target_corr = np.abs(covariance_xy / (np.sqrt(variance_x * variance_y) + 1e-30))
    covariance_x = cross - np.outer(sum_x, sum_x) / rows
    denominator = np.sqrt(variance_x[:, None] * variance_x[None, :])
    correlation = np.abs(covariance_x / (denominator + 1e-30))
    return target_corr, correlation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["late", "first"], default="late")
    args = parser.parse_args()
    features, targets, names = ensure_our379_cache()
    sample_ids = np.load(SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(SOURCE_CACHE / "months.npy", mmap_mode="r")
    reference.absolute_correlations = chunked_absolute_correlations
    reference.RUN_DIR = RUN_DIR
    fold_settings = {
        "late": ("train059_valid6270_ex66", 59, 62, 70, YUNSU_LATE_REFERENCE),
        "first": ("train049_valid5059", 49, 50, 59, YUNSU_FIRST_REFERENCE),
    }
    fold_name, train_end, valid_start, valid_end, yunsu_reference = fold_settings[args.fold]
    result = reference.run_fold(
        fold_name,
        train_end,
        valid_start,
        valid_end,
        features,
        targets,
        names,
        sample_ids,
        months,
        10,
        False,
    )
    score = float(result["best_validation_cosine"])
    summary = {
        "experiment": "EXP-REALMLP-006-OUR379-RQ-REFERENCE",
        "model": "Yunsu public RQ RealMLP",
        "only_changed": "Yunsu public152 features -> our379 features",
        "fold": result["fold"],
        "best_epoch": result["best_epoch"],
        "best_validation_cosine": score,
        "yunsu_public_feature_reference": yunsu_reference,
        "delta": score - yunsu_reference,
        "passed_replacement_gate": score > yunsu_reference,
        "kept_features": result["selection"]["kept_features"],
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"{fold_name}_score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPARISON " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
