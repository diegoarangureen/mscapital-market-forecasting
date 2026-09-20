"""Test RQ RealMLP379 without the 100-feature correlation pruning step."""
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


RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-008-OUR379-ALL-FEATURES"


def select_all_features(sample_ids, features, target, feature_names):
    del sample_ids, features, target
    feature_count = len(feature_names)
    return (
        np.arange(feature_count, dtype=np.int64),
        list(feature_names),
        {
            "input_features": feature_count,
            "kept_features": feature_count,
            "dropped_features": [],
            "sample_id_dropped": True,
            "high_correlation_pairs_processed": 0,
        },
    )


def main() -> None:
    features, targets, names = source.ensure_our379_cache()
    sample_ids = np.load(source.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(source.SOURCE_CACHE / "months.npy", mmap_mode="r")
    reference.select_features = select_all_features
    reference.RUN_DIR = RUN_DIR
    result = reference.run_fold(
        "train059_valid6270_ex66",
        59,
        62,
        70,
        features,
        targets,
        names,
        sample_ids,
        months,
        10,
        False,
    )
    summary = {
        "experiment": "EXP-REALMLP-008-OUR379-ALL-FEATURES",
        "only_changed": "keep all 379 features instead of correlation-pruned 279",
        "baseline_score": 0.15348300645832985,
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
