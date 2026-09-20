"""Test a stronger metric-aligned loss on the validated RQ RealMLP379."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import exp_realmlp_005_yunsu_public_reference as reference
import exp_realmlp_006_our379_rq_reference as source


RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-007-OUR379-RQ-COSINE05"


def compute_loss_cosine05(
    prediction: torch.Tensor,
    target: torch.Tensor,
    code_logits: list[torch.Tensor],
    codes: torch.Tensor,
    rq_weight: float,
):
    prediction = prediction.squeeze(-1)
    repeated_target = target[:, None].expand_as(prediction)
    prediction_flat = prediction.reshape(-1)
    target_flat = repeated_target.reshape(-1)
    sample_weight = torch.where(target_flat.abs() > 0.001, 0.5, 1.0)
    mse = (sample_weight * (prediction_flat - target_flat).square()).mean()
    pred_centered = prediction_flat - prediction_flat.mean()
    target_centered = target_flat - target_flat.mean()
    cosine = F.cosine_similarity(pred_centered, target_centered, dim=0)
    rq_loss = torch.zeros((), device=prediction.device)
    expanded_codes = codes[:, None].expand(-1, reference.N_ENSEMBLE, -1)
    for layer, logits in enumerate(code_logits):
        labels = expanded_codes[:, :, layer].reshape(-1)
        rq_loss = rq_loss + F.cross_entropy(logits.reshape(-1, 3), labels)
    rq_loss = rq_loss / len(code_logits)
    return mse + 0.05 * (1.0 - cosine) + rq_weight * rq_loss, cosine, mse, rq_loss


def main() -> None:
    features, targets, names = source.ensure_our379_cache()
    sample_ids = source.np.load(source.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = source.np.load(source.SOURCE_CACHE / "months.npy", mmap_mode="r")
    reference.absolute_correlations = source.chunked_absolute_correlations
    reference.compute_loss = compute_loss_cosine05
    reference.RUN_DIR = RUN_DIR
    result = reference.run_fold(
        "train059_valid6270_ex66",
        59,
        62,
        70,
        features,
        targets,
        names,
        sample_ids,
        months,
        10,
        False,
    )
    summary = {
        "experiment": "EXP-REALMLP-007-OUR379-RQ-COSINE05",
        "only_changed": "centered cosine loss coefficient 0.01 -> 0.05",
        "baseline_score": 0.15348300645832985,
        "best_epoch": result["best_epoch"],
        "best_validation_cosine": result["best_validation_cosine"],
        "delta": result["best_validation_cosine"] - 0.15348300645832985,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPARISON " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
