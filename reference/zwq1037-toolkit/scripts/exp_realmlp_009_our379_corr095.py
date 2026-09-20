"""Test a gentler 0.95 correlation-pruning threshold for RQ RealMLP379."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import exp_realmlp_005_yunsu_public_reference as reference
import exp_realmlp_006_our379_rq_reference as source


RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095"


def select_features_corr095(train_ids, train_features, train_target, feature_names):
    correlation_input = np.column_stack((train_ids.astype(np.float64), train_features))
    all_names = ["sample_id", *feature_names]
    target_corr, corr = source.chunked_absolute_correlations(correlation_input, train_target)
    upper_i, upper_j = np.triu_indices(corr.shape[0], k=1)
    pair_values = corr[upper_i, upper_j]
    high = np.flatnonzero(pair_values >= 0.95)
    order = high[np.argsort(pair_values[high])[::-1]]
    dropped = set()
    pairs = []
    for position in order:
        left = int(upper_i[position])
        right = int(upper_j[position])
        if left in dropped or right in dropped:
            continue
        dropped.add(right if target_corr[left] >= target_corr[right] else left)
        pairs.append((left, right, float(pair_values[position])))
    standard_deviation = np.std(correlation_input, axis=0)
    for index in range(correlation_input.shape[1]):
        if standard_deviation[index] == 0.0 or target_corr[index] < 0.0001:
            dropped.add(index)
    keep = np.asarray(
        [index - 1 for index in range(1, len(all_names)) if index not in dropped],
        dtype=np.int64,
    )
    kept_names = [feature_names[index] for index in keep]
    details = {
        "input_features": len(feature_names),
        "kept_features": len(kept_names),
        "dropped_features": [all_names[index] for index in sorted(dropped) if index > 0],
        "sample_id_dropped": 0 in dropped,
        "high_correlation_pairs_processed": len(pairs),
        "correlation_threshold": 0.95,
    }
    return keep, kept_names, details


def main() -> None:
    features, targets, names = source.ensure_our379_cache()
    sample_ids = np.load(source.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(source.SOURCE_CACHE / "months.npy", mmap_mode="r")
    reference.select_features = select_features_corr095
    reference.RUN_DIR = RUN_DIR
    result = reference.run_fold(
        "train059_valid6270_ex66", 59, 62, 70,
        features, targets, names, sample_ids, months, 10, False,
    )
    summary = {
        "experiment": "EXP-REALMLP-009-OUR379-CORR095",
        "only_changed": "correlation pruning threshold 0.90 -> 0.95",
        "baseline_score": 0.15348300645832985,
        "kept_features": result["selection"]["kept_features"],
        "best_epoch": result["best_epoch"],
        "best_validation_cosine": result["best_validation_cosine"],
        "delta": result["best_validation_cosine"] - 0.15348300645832985,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPARISON " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
