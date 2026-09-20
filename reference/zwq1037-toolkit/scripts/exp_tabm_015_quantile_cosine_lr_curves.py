"""Repeat EXP-TABM-014 with a smooth cosine learning-rate schedule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import exp_tabm_014_quantile_stage_curves as experiment


EXPERIMENT_ID = "EXP-TABM-015-QUANTILE-COSINE-LR-CURVES"
INITIAL_LEARNING_RATE = 2.0e-3
MINIMUM_LEARNING_RATE = 1.0e-4
BASE_TRAIN_ONE_EPOCH = experiment.tabm.train_one_epoch


def learning_rate_at(epoch: int) -> float:
    fraction = (epoch - 1) / max(experiment.MAX_EPOCHS - 1, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return MINIMUM_LEARNING_RATE + (
        INITIAL_LEARNING_RATE - MINIMUM_LEARNING_RATE
    ) * cosine_factor


def scheduled_train_one_epoch(
    model,
    optimizer,
    features,
    targets_scaled,
    train_indices,
    preprocessor,
    epoch,
):
    learning_rate = learning_rate_at(epoch)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return BASE_TRAIN_ONE_EPOCH(
        model,
        optimizer,
        features,
        targets_scaled,
        train_indices,
        preprocessor,
        epoch,
    )


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.tabm.LEARNING_RATE = INITIAL_LEARNING_RATE
    experiment.tabm.train_one_epoch = scheduled_train_one_epoch
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    for fold_name in experiment.FOLDS:
        curve_path = run_dir / fold_name / "epoch_curve.csv"
        curve = pd.read_csv(curve_path)
        curve.insert(
            2,
            "learning_rate",
            [learning_rate_at(int(epoch)) for epoch in curve["epoch"]],
        )
        curve.to_csv(curve_path, index=False)
        result_path = run_dir / fold_name / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["learning_rate"] = {
            "schedule": "smooth cosine decay by epoch",
            "initial": INITIAL_LEARNING_RATE,
            "minimum": MINIMUM_LEARNING_RATE,
            "length_epochs": experiment.MAX_EPOCHS,
        }
        result.pop("learning_rate_schedule", None)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    top_path = run_dir / "result.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    top["learning_rate"] = {
        "schedule": "smooth cosine decay by epoch",
        "initial": INITIAL_LEARNING_RATE,
        "minimum": MINIMUM_LEARNING_RATE,
        "length_epochs": experiment.MAX_EPOCHS,
    }
    top_path.write_text(
        json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
