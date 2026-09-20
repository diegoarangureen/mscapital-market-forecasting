"""Paired development comparison for kernel-5 Conv-GRU across seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import cosine


def score_pair(baseline_path: Path, candidate_path: Path) -> dict[str, float]:
    baseline = pd.read_feather(baseline_path)
    candidate = pd.read_feather(candidate_path)
    if not np.array_equal(baseline["sample_id"], candidate["sample_id"]):
        raise AssertionError("Paired GRU sample IDs differ.")
    if not np.array_equal(baseline["month"], candidate["month"]):
        raise AssertionError("Paired GRU months differ.")
    if not np.allclose(baseline["target"], candidate["target"]):
        raise AssertionError("Paired GRU targets differ.")

    target = baseline["target"].to_numpy(dtype=np.float64)
    scores = {}
    for name, column in (("standalone", "prediction"), ("blend90", "blend90")):
        before = cosine(
            target, baseline[column].to_numpy(dtype=np.float64)
        )
        after = cosine(
            target, candidate[column].to_numpy(dtype=np.float64)
        )
        scores[name] = {
            "joint_gru": before,
            "conv_gru": after,
            "delta": after - before,
        }
    return scores


def main() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "data" / "interim" / "sequence_experiments"
    )
    sources = {
        "42": (
            root / "EXP-GRU-003-STRONG-JOINT-DEV"
            / "joint_gru319" / "validation_predictions_epoch06.feather",
            root / "EXP-GRU-006-CONV-JOINT-DEV"
            / "joint_conv_gru319" / "validation_predictions_epoch06.feather",
        ),
        "137": (
            root / "EXP-GRU-005-JOINT-SEED137"
            / "train049_valid5059" / "joint_gru319"
            / "validation_predictions_epoch06.feather",
            root / "EXP-GRU-008-CONV-JOINT-SEED137-DEV"
            / "joint_conv_gru319" / "validation_predictions_epoch06.feather",
        ),
    }
    by_seed = {
        seed: score_pair(*paths)
        for seed, paths in sources.items()
    }
    deltas = [
        by_seed[seed]["standalone"]["delta"]
        for seed in ("42", "137")
    ]
    result = {
        "experiment": "EXP-GRU-008-CONV-JOINT-SEED137-DEV",
        "train_months": "0-49",
        "validation_months": "50-59",
        "by_seed": by_seed,
        "mean_standalone_delta": float(np.mean(deltas)),
        "advance_if_mean_at_least_0_0005_and_both_positive": (
            all(delta > 0 for delta in deltas)
            and float(np.mean(deltas)) >= 0.0005
        ),
    }
    output_path = (
        root / "EXP-GRU-008-CONV-JOINT-SEED137-DEV"
        / "paired_dev_comparison.json"
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
