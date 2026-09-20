"""Test fixed 24-month recency weighting for Relative319 TabM MSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_016_relative_scale_features as experiment


HALF_LIFE_MONTHS = 24.0
ROW_MONTHS: np.ndarray | None = None


def train_one_epoch_recency(
    model,
    optimizer,
    features: np.ndarray,
    targets_scaled: np.ndarray,
    train_indices: np.ndarray,
    preprocessor,
    epoch: int,
) -> float:
    """较近月份得到较大权重。 / Give more recent months more weight."""
    if ROW_MONTHS is None or len(ROW_MONTHS) != len(features):
        raise AssertionError("Training months and feature rows are not aligned.")
    latest_month = int(ROW_MONTHS[train_indices].max())
    train_weights = np.exp2(
        (ROW_MONTHS[train_indices].astype(np.float32) - latest_month)
        / HALF_LIFE_MONTHS
    )
    train_weights /= train_weights.mean(dtype=np.float64)
    row_weights = np.zeros(len(features), dtype=np.float32)
    row_weights[train_indices] = train_weights

    model.train()
    shuffled_indices = np.random.default_rng(
        experiment.tabm.SEED + epoch
    ).permutation(train_indices)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(shuffled_indices), experiment.tabm.BATCH_SIZE):
        batch_indices = shuffled_indices[
            start : start + experiment.tabm.BATCH_SIZE
        ]
        batch = preprocessor.transform(features[batch_indices])
        target = torch.as_tensor(
            targets_scaled[batch_indices],
            dtype=torch.float32,
            device=preprocessor.device,
        )
        weights = torch.as_tensor(
            row_weights[batch_indices],
            dtype=torch.float32,
            device=preprocessor.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            member_predictions = model(batch).squeeze(-1).float()
        squared_error = (
            member_predictions - target[:, None]
        ).square()
        loss = (squared_error * weights[:, None]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count = len(batch_indices)
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
    return total_loss / total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 137), required=True)
    parser.add_argument(
        "--stage", choices=("dev", "confirm"), required=True
    )
    args = parser.parse_args()
    fold = (
        "train049_valid5059"
        if args.stage == "dev"
        else "train059_valid6070"
    )
    fold_months = (
        (49, 50, 59)
        if args.stage == "dev"
        else (59, 60, 70)
    )
    experiment_id = (
        f"EXP-TABM-024-RECENCY24-SEED{args.seed}-{args.stage.upper()}"
    )
    project_dir = Path(__file__).resolve().parents[1]
    export_dir = project_dir / "data" / "interim" / "kaggle_relative319_dev"
    exported_ids = np.load(export_dir / "sample_ids.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month"],
    ).sort_values("sample_id")
    if not np.array_equal(
        labels["sample_id"].to_numpy(), exported_ids
    ):
        raise AssertionError("Month labels do not match Relative319 feature rows.")
    global ROW_MONTHS
    ROW_MONTHS = labels["month"].to_numpy(dtype=np.int16, copy=True)

    experiment.EXPERIMENT_ID = experiment_id
    experiment.SEED = args.seed
    experiment.FOLDS = {fold: fold_months}
    experiment.tabm.train_one_epoch = train_one_epoch_recency
    experiment.main()

    root = project_dir / "data" / "interim" / "tree_experiments"
    candidate_path = root / experiment_id / "result.json"
    candidate_payload = json.loads(
        candidate_path.read_text(encoding="utf-8")
    )
    candidate_payload.update(
        {
            "single_variable_change": (
                "24-month exponential recency weighting of per-member MSE"
            ),
            "baseline": "same-seed Relative319 uniform-weight MSE",
            "baseline_feature_count": 319,
            "candidate_feature_count": 319,
            "added_features": [],
            "half_life_months": HALF_LIFE_MONTHS,
            "row_weight_mean_normalized_to": 1.0,
        }
    )
    candidate_path.write_text(
        json.dumps(candidate_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    candidate = candidate_payload["folds"][fold]["metrics"]
    if args.seed == 42:
        baseline = json.loads(
            (root / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / "result.json")
            .read_text(encoding="utf-8")
        )["folds"][fold]["metrics"]
    else:
        baseline = json.loads(
            (root / "EXP-TABM-019-RELATIVE-ADD12-SEED137-PAIRED"
             / "result.json").read_text(encoding="utf-8")
        )["results"]["relative_add12"][fold]["metrics"]
    comparison = {
        "experiment_id": experiment_id,
        "seed": args.seed,
        "stage": args.stage,
        "fold": fold,
        "half_life_months": HALF_LIFE_MONTHS,
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
    (root / experiment_id / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
