"""Run official TabM with per-member batch cosine loss and cosine LR decay."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as experiment


EXPERIMENT_ID = "EXP-TABM-003-COSINE"
INITIAL_LEARNING_RATE = 1.0e-3
MINIMUM_LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-5
MAX_EPOCHS = 30


def current_learning_rate(epoch: int) -> float:
    fraction = (epoch - 1) / max(MAX_EPOCHS - 1, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return MINIMUM_LEARNING_RATE + (
        INITIAL_LEARNING_RATE - MINIMUM_LEARNING_RATE
    ) * cosine_factor


def train_one_epoch_cosine(
    model: experiment.TabM,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    targets_scaled: np.ndarray,
    train_indices: np.ndarray,
    preprocessor: experiment.BatchPreprocessor,
    epoch: int,
) -> float:
    """每个并行成员独立计算批内 cosine loss。 / Train each member with cosine loss."""
    learning_rate = current_learning_rate(epoch)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate

    model.train()
    generator = np.random.default_rng(experiment.SEED + epoch)
    shuffled_indices = generator.permutation(train_indices)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(shuffled_indices), experiment.BATCH_SIZE):
        batch_indices = shuffled_indices[start : start + experiment.BATCH_SIZE]
        batch = preprocessor.transform(features[batch_indices])
        target = torch.as_tensor(
            targets_scaled[batch_indices],
            dtype=torch.float32,
            device=preprocessor.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            member_predictions = model(batch).squeeze(-1).float()

        centered_predictions = member_predictions - member_predictions.mean(
            dim=0, keepdim=True
        )
        centered_target = target - target.mean()
        numerator = (centered_predictions * centered_target[:, None]).sum(dim=0)
        prediction_norm = torch.linalg.vector_norm(centered_predictions, dim=0)
        target_norm = torch.linalg.vector_norm(centered_target)
        member_cosines = numerator / (prediction_norm * target_norm + 1.0e-8)
        loss = 1.0 - member_cosines.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count = int(len(batch_indices))
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
    print(f"epoch={epoch:02d} learning_rate={learning_rate:.8f}", flush=True)
    return total_loss / total_count


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.SEED = 42
    experiment.LEARNING_RATE = INITIAL_LEARNING_RATE
    experiment.WEIGHT_DECAY = WEIGHT_DECAY
    experiment.MAX_SCOUT_EPOCHS = MAX_EPOCHS
    experiment.PATIENCE = 5
    experiment.train_one_epoch = train_one_epoch_cosine
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
        / "config.json"
    )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    metadata["parameters"].update(
        {
            "loss": "mean per-member batch cosine loss",
            "learning_rate_schedule": "cosine decay",
            "initial_learning_rate": INITIAL_LEARNING_RATE,
            "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            "schedule_length_epochs": MAX_EPOCHS,
        }
    )
    metadata["public_notebook_idea"] = (
        "cosine loss inspired by local copy of kgpu-tabm-cos689-3seed; "
        "implemented on official tabm package rather than its custom MLP"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
