"""Add a zero-initialized memberwise numerical skip to frozen RQ RealMLP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import torch
from torch import nn

import exp_realmlp_005_yunsu_public_reference as reference
import exp_realmlp_006_our379_rq_reference as source
import exp_realmlp_009_our379_corr095 as selection

PROJECT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT / "data/interim/tree_experiments/EXP-REALMLP-010-NUMERICAL-SKIP"
BaseRealMLPRQ = reference.RealMLPRQ


class NumericalSkipRealMLPRQ(BaseRealMLPRQ):
    def __init__(self, n_numerical: int, cat_dims: list[int]):
        super().__init__(n_numerical, cat_dims)
        # 零初始化使初始预测与基线完全相同，之后学习简单数值关系。
        # Zero initialization preserves the baseline prediction at startup.
        self.numerical_skip_weight = nn.Parameter(
            torch.zeros(reference.N_ENSEMBLE, n_numerical, 1)
        )

    def forward(self, numerical, categorical, return_codes=False):
        codes, regression = super().forward(numerical, categorical, return_codes=True)
        skip = torch.einsum("bi,nio->bno", numerical, self.numerical_skip_weight)
        regression = regression + skip / np.sqrt(numerical.shape[1])
        if return_codes:
            return codes, regression
        return regression.mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(2)
    features, targets, names = source.ensure_our379_cache()
    ids = np.load(source.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.load(source.SOURCE_CACHE / "months.npy", mmap_mode="r")
    reference.select_features = selection.select_features_corr095
    reference.RealMLPRQ = NumericalSkipRealMLPRQ
    reference.RUN_DIR = RUN_DIR / "smoke" if args.smoke else RUN_DIR
    result = reference.run_fold(
        "train059_valid6270_ex66", 59, 62, 70, features, targets, names,
        ids, months, args.epochs, args.smoke,
    )
    baseline = 0.15396996342598498
    summary = {
        "experiment": "EXP-REALMLP-010-NUMERICAL-SKIP",
        "change": "zero initialized numerical memberwise prediction skip only",
        "baseline_centered_cosine": baseline,
        "candidate_centered_cosine": result["best_validation_cosine"],
        "delta": result["best_validation_cosine"] - baseline,
        "best_epoch": result["best_epoch"],
        "epochs": args.epochs,
        "smoke": args.smoke,
    }
    (reference.RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
