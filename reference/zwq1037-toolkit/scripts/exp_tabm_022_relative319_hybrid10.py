"""Test a modest cosine term in the Relative319 TabM MSE objective."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import exp_tabm_016_relative_scale_features as experiment


EXPERIMENT_ID = "EXP-TABM-022-RELATIVE319-HYBRID10"
DEV_FOLD = "train049_valid5059"
MSE_WEIGHT = 0.90
COSINE_WEIGHT = 0.10


def train_one_epoch_hybrid(
    model,
    optimizer,
    features: np.ndarray,
    targets_scaled: np.ndarray,
    train_indices: np.ndarray,
    preprocessor,
    epoch: int,
) -> float:
    """以 MSE 为主，少量加入 cosine。 / Add a small cosine term to MSE."""
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
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(batch).squeeze(-1).float()

        mse_loss = F.mse_loss(
            prediction, target[:, None].expand_as(prediction)
        )
        centered_prediction = prediction - prediction.mean(
            dim=0, keepdim=True
        )
        centered_target = target - target.mean()
        numerator = (
            centered_prediction * centered_target[:, None]
        ).sum(dim=0)
        denominator = (
            torch.linalg.vector_norm(centered_prediction, dim=0)
            * torch.linalg.vector_norm(centered_target)
        )
        cosine_loss = 1.0 - (numerator / (denominator + 1.0e-8)).mean()
        loss = MSE_WEIGHT * mse_loss + COSINE_WEIGHT * cosine_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count = len(batch_indices)
        total_loss += float(loss.detach().cpu()) * count
        total_count += count

    return total_loss / total_count


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.FOLDS = {
        DEV_FOLD: (49, 50, 59),
        'train059_valid6070': (59, 60, 70),
    }
    experiment.tabm.train_one_epoch = train_one_epoch_hybrid
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    root = project_dir / "data" / "interim" / "tree_experiments"
    candidate = json.loads(
        (root / EXPERIMENT_ID / "result.json").read_text(encoding="utf-8")
    )["folds"][DEV_FOLD]["metrics"]
    baseline = json.loads(
        (root / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / "result.json")
        .read_text(encoding="utf-8")
    )["folds"][DEV_FOLD]["metrics"]
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "single_variable_change": "loss = 0.9 MSE + 0.1 batch cosine",
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

