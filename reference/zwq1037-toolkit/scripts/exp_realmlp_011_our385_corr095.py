"""Single-change test: add six market trajectory features to RQ RealMLP379."""

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
RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-011-OUR385-CORR095"
CACHE = PROJECT / "data/interim/our385_realmlp_reference_cache"
TRAJECTORY = (
    "mid_return_180 mid_return_60 mid_return_20 "
    "realized_volatility_20 realized_volatility_60 mid_momentum_60"
).split()
FOLD = "train059_valid6270_ex66"


def ensure_cache():
    base, target, names = featuresource.ensure_our379_cache()
    sample_ids = np.load(featuresource.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    path = CACHE / "features.npy"
    ready = CACHE / "READY.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    if ready.exists() and path.exists():
        output = np.load(path, mmap_mode="r")
        if output.shape != (len(sample_ids), 385):
            raise AssertionError("Unexpected cached shape")
        return output, target, names + TRAJECTORY

    # 只读取现有六列，不重新聚合原始事件。
    # Read six existing columns rather than aggregating raw events again.
    state = pd.read_feather(
        PROJECT / "data/processed/train_market_microstructure_features.feather",
        columns=["sample_id", *TRAJECTORY],
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(state["sample_id"].to_numpy(), sample_ids):
        raise AssertionError("Trajectory IDs differ from reference cache")
    values = state[TRAJECTORY].to_numpy(dtype=np.float32)
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(sample_ids), 385)
    )
    for start in range(0, len(sample_ids), 8192):
        stop = min(start + 8192, len(sample_ids))
        output[start:stop, :379] = base[start:stop]
        output[start:stop, 379:] = values[start:stop]
    output.flush()
    ready.write_text(json.dumps({"rows": len(sample_ids), "columns": names + TRAJECTORY},
                               indent=2), encoding="utf-8")
    del output, state, values
    gc.collect()
    return np.load(path, mmap_mode="r"), target, names + TRAJECTORY


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
    features, target, names = ensure_cache()
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
    summary = {"experiment": "EXP-REALMLP-011-OUR385-CORR095",
               "only_changed": "379 -> 385 features; structure/loss/seed/epochs unchanged",
               "fold": "train0-59, purge60-61, valid62-70 excluding66",
               "best_epoch": result["best_epoch"],
               "kept_features": result["selection"]["kept_features"],
               "new_features_kept": [name for name in TRAJECTORY if name in result["kept_feature_names"]],
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
