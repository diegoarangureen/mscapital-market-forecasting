"""Single-change test: halve RQ auxiliary loss weight on RealMLP379."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch

import exp_realmlp_006_our379_rq_reference as featuresource
import exp_realmlp_009_our379_corr095 as selector
import realmlp011_reference as runner

PROJECT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-012-OUR379-RQHALF"
FOLD = "train059_valid6270_ex66"

# 只减半 RQ 辅助损失，保留原有退火和预测损失。
# Halve only the RQ auxiliary weight; preserve annealing and prediction loss.
_original_compute_loss = runner.compute_loss

def compute_loss_rq_half(prediction, target, code_logits, codes, rq_weight):
    return _original_compute_loss(prediction, target, code_logits, codes, rq_weight * 0.5)


def raw_cosine(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.dot(target, prediction) / (np.linalg.norm(target) * np.linalg.norm(prediction)))


def main() -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    runner.RUN_DIR = RUN_DIR
    runner.select_features = selector.select_features_corr095
    runner.compute_loss = compute_loss_rq_half
    features, target, names = featuresource.ensure_our379_cache()
    sample_ids = np.load(featuresource.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(featuresource.SOURCE_CACHE / "months.npy", mmap_mode="r")
    print(f"GPU={torch.cuda.get_device_name(0)} features={features.shape} CPU_threads=2", flush=True)
    result_path = RUN_DIR / FOLD / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = runner.run_fold(
            FOLD, 59, 62, 70, features, target, names, sample_ids, months, 10, False
        )

    baseline_dir = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095" / FOLD
    ids = np.load(RUN_DIR / FOLD / "validation_sample_ids.npy")
    baseline_ids = np.load(baseline_dir / "validation_sample_ids.npy")
    if not np.array_equal(ids, baseline_ids):
        raise AssertionError("Baseline and candidate validation IDs differ")
    truth = np.load(RUN_DIR / FOLD / "validation_targets.npy")
    old = np.load(baseline_dir / "validation_predictions.npy")
    new = np.load(RUN_DIR / FOLD / "validation_predictions.npy")
    valid_rows = np.flatnonzero((months >= 62) & (months <= 70) & (months != 66))
    valid_months = np.asarray(months[valid_rows])
    frame = pd.DataFrame({"sample_id": ids, "month": valid_months, "target": truth,
                          "baseline": old, "prediction": new})
    frame.to_feather(RUN_DIR / FOLD / "validation_predictions.feather")
    metrics = {}
    for name, mask in [("all", np.ones(len(ids), dtype=bool)),
                       ("62_65", np.isin(valid_months, [62, 63, 64, 65])),
                       ("67_70", np.isin(valid_months, [67, 68, 69, 70]))]:
        baseline_raw = raw_cosine(truth[mask], old[mask])
        candidate_raw = raw_cosine(truth[mask], new[mask])
        metrics[name] = {"baseline_raw_cosine": baseline_raw,
                         "candidate_raw_cosine": candidate_raw,
                         "raw_delta": candidate_raw - baseline_raw,
                         "baseline_centered": runner.cosine_score(truth[mask], old[mask]),
                         "candidate_centered": runner.cosine_score(truth[mask], new[mask])}
    summary = {"experiment": "EXP-REALMLP-012-OUR379-RQHALF",
               "only_changed": "RQ auxiliary loss multiplier 0.5; 379 features/structure/seed/epochs unchanged",
               "fold": "train0-59, purge60-61, valid62-70 excluding66",
               "best_epoch": result["best_epoch"],
               "kept_features": result["selection"]["kept_features"],
               "rq_loss_multiplier": 0.5,
               "metrics": metrics,
               "per_month_raw_delta": {str(int(month)): raw_cosine(truth[valid_months == month], new[valid_months == month])
                                        - raw_cosine(truth[valid_months == month], old[valid_months == month])
                                        for month in np.unique(valid_months)},
               "passed": metrics["all"]["raw_delta"] >= 0.0007
                         and metrics["62_65"]["raw_delta"] > 0
                         and metrics["67_70"]["raw_delta"] > 0}
    for filename in ("score_summary.json", "score_only.json"):
        (RUN_DIR / filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
