"""Compare per-member batch cosine loss with the Relative319 TabM MSE baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import exp_tabm_016_relative_scale_features as experiment


EXPERIMENT_ID = "EXP-TABM-021-RELATIVE319-COSINE-LOSS"
DEV_FOLD = "train049_valid5059"


def train_one_epoch_cosine(
    model,
    optimizer,
    features: np.ndarray,
    targets_scaled: np.ndarray,
    train_indices: np.ndarray,
    preprocessor,
    epoch: int,
) -> float:
    """逐成员优化批内相关方向。 / Optimize batch cosine for each TabM member."""
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
            member_predictions = model(batch).squeeze(-1).float()

        # 先去掉批内均值，再比较预测与目标的方向。
        # Center each batch before comparing prediction and target directions.
        centered_predictions = member_predictions - member_predictions.mean(
            dim=0, keepdim=True
        )
        centered_target = target - target.mean()
        numerator = (
            centered_predictions * centered_target[:, None]
        ).sum(dim=0)
        prediction_norm = torch.linalg.vector_norm(
            centered_predictions, dim=0
        )
        target_norm = torch.linalg.vector_norm(centered_target)
        member_cosine = numerator / (
            prediction_norm * target_norm + 1.0e-8
        )
        loss = 1.0 - member_cosine.mean()
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
    experiment.tabm.train_one_epoch = train_one_epoch_cosine
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments"
    candidate_path = run_dir / EXPERIMENT_ID / "result.json"
    baseline_path = (
        run_dir / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / "result.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_metrics = candidate["folds"][DEV_FOLD]["metrics"]
    baseline_metrics = baseline["folds"][DEV_FOLD]["metrics"]
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "single_variable_change": "per-member batch cosine instead of per-member MSE",
        "baseline": "EXP-TABM-016 Relative319 seed42 epoch15",
        "train_months": "0-49",
        "validation_months": "50-59",
        "tabm_mean_delta": (
            candidate_metrics["tabm_mean"]["overall"]
            - baseline_metrics["tabm_mean"]["overall"]
        ),
        "tabm_trim1_delta": (
            candidate_metrics["tabm_trim1"]["overall"]
            - baseline_metrics["tabm_trim1"]["overall"]
        ),
        "b001_trim1_delta": (
            candidate_metrics["b001_trim1"]["overall"]
            - baseline_metrics["b001_trim1"]["overall"]
        ),
    }
    (run_dir / EXPERIMENT_ID / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

