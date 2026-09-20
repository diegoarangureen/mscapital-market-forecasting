"""Screen structural candidates with aligned saved validation predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT / "data/interim/tree_experiments"


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x)


def score(y, p):
    return float(np.dot(y, p) / (np.linalg.norm(y) * np.linalg.norm(p)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-dir", type=Path)
    args = parser.parse_args()
    tabm = pd.read_feather(EXPERIMENTS / "EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather")
    tabm = tabm.loc[tabm.month.ne(66), ["sample_id", "month", "target", "candidate"]].rename(columns={"candidate": "tabm"})
    transformer = pd.read_csv(PROJECT / "outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv")
    transformer = transformer[["sample_id", "prediction"]].rename(columns={"prediction": "transformer"})
    real_dir = EXPERIMENTS / "EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66"
    real = pd.DataFrame({"sample_id": np.load(real_dir / "validation_sample_ids.npy"), "realmlp": np.load(real_dir / "validation_predictions.npy")})
    frame = tabm.merge(transformer, on="sample_id", validate="one_to_one").merge(real, on="sample_id", validate="one_to_one")
    candidates = {}
    real_new_dir = EXPERIMENTS / "EXP-REALMLP-010-NUMERICAL-SKIP/train059_valid6270_ex66"
    if (real_new_dir / "result.json").exists():
        candidates["realmlp_skip"] = ("realmlp", pd.DataFrame({"sample_id": np.load(real_new_dir / "validation_sample_ids.npy"), "prediction": np.load(real_new_dir / "validation_predictions.npy")}))
    if args.transformer_dir:
        candidates["transformer_multiwindow10"] = ("transformer", pd.read_csv(args.transformer_dir / "validation_predictions.csv")[["sample_id", "prediction"]])
    if len(frame) != 140806:
        raise AssertionError("Baseline validation IDs do not align.")
    target = frame.target.to_numpy(dtype=np.float64)
    months = frame.month.to_numpy()
    # Public models lack matching local validation; this is an owned-block proxy.
    weights = {"tabm": 0.12 / 0.33, "realmlp": 0.03 / 0.33, "transformer": 0.18 / 0.33}
    normalized = {name: unit(frame[name]) for name in weights}
    base = sum(weights[name] * normalized[name] for name in weights)
    rows = []
    for name, (member, predictions) in candidates.items():
        aligned = frame[["sample_id"]].merge(predictions, on="sample_id", validate="one_to_one")
        if len(aligned) != len(frame):
            raise AssertionError(f"Candidate IDs differ: {name}")
        raw = aligned.prediction.to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all():
            raise AssertionError(f"Nonfinite candidate: {name}")
        candidate = unit(raw)
        row = {"name": name, "member": member, "baseline_raw_cosine": score(target, normalized[member]), "candidate_raw_cosine": score(target, candidate), "single_delta": score(target, candidate) - score(target, normalized[member]), "baseline_candidate_correlation": float(np.corrcoef(frame[member], raw)[0, 1]), "replacements": []}
        for fraction in (0.5, 1.0):
            prediction = base + weights[member] * fraction * (candidate - normalized[member])
            row["replacements"].append({"fraction": fraction, "owned_proxy_cosine": score(target, prediction), "delta": score(target, prediction) - score(target, base), "monthly_delta": {str(int(month)): score(target[months == month], prediction[months == month]) - score(target[months == month], base[months == month]) for month in np.unique(months)}})
        rows.append(row)
    report = {"rows": len(frame), "window": "train0-59 purge60-61 valid62-70 no66", "baseline_owned_proxy": score(target, base), "proxy_weights": weights, "limitation": "Public block and temporal-GRU matching validation predictions unavailable; this proxy is not the exact 0.151 ensemble. RealMLP skip is paired against strict 009 baseline, not production fold57.", "candidates": rows}
    output = PROJECT / "outputs/submission_metadata/structure_v13_realmlp010_screen.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
