"""Paired seed-137 test of the Relative319 TabM 90/10 hybrid objective."""

from __future__ import annotations

import json
from pathlib import Path

import exp_tabm_016_relative_scale_features as experiment
from exp_tabm_022_relative319_hybrid10 import train_one_epoch_hybrid


EXPERIMENT_ID = "EXP-TABM-023-RELATIVE319-HYBRID10-SEED137"
BASELINE_ID = "EXP-TABM-019-RELATIVE-ADD12-SEED137-PAIRED"
DEV_FOLD = "train049_valid5059"


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.SEED = 137
    experiment.FOLDS = {DEV_FOLD: (49, 50, 59)}
    experiment.tabm.train_one_epoch = train_one_epoch_hybrid
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    root = project_dir / "data" / "interim" / "tree_experiments"
    candidate = json.loads(
        (root / EXPERIMENT_ID / "result.json").read_text(encoding="utf-8")
    )["folds"][DEV_FOLD]["metrics"]
    baseline = json.loads(
        (root / BASELINE_ID / "result.json").read_text(encoding="utf-8")
    )["results"]["relative_add12"][DEV_FOLD]["metrics"]
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "baseline": BASELINE_ID,
        "seed": 137,
        "loss": "0.9 MSE + 0.1 batch cosine",
        "train_months": "0-49",
        "validation_months": "50-59",
        "tabm_mean_delta": (
            candidate["tabm_mean"]["overall"]
            - baseline["tabm_mean"]["overall"]
        ),
        "tabm_trim1_delta": (
            candidate["tabm_trim1"]["overall"]
            - baseline["tabm_trim1"]["overall"]
        ),
        "b001_trim1_delta": (
            candidate["b001_trim1"]["overall"]
            - baseline["b001_trim1"]["overall"]
        ),
    }
    (root / EXPERIMENT_ID / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
