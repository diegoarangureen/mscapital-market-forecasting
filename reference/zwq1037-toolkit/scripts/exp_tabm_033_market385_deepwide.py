"""Validate a public-inspired deeper and wider official TabM on 385 features."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = (
    PROJECT_DIR
    / "data"
    / "interim"
    / "kaggle_kernels"
    / "relative319_xs_tabm_notebook"
)
sys.path.insert(0, str(REFERENCE_DIR))

import run_extracted as recipe
import exp_gru_003_strong_joint as gru
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_COLUMNS


MARKET_COLUMNS = (
    "mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 "
    "realized_volatility_60 mid_momentum_60"
).split()
EXPERIMENT_ID = "EXP-TABM-033-MARKET385-DEEPWIDE"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
FOLD_NAME = "train059_valid6270_ex66"
SCORE_KEY = "months_62_70_without_66"


def compact_result(result: dict) -> dict:
    """Remove arrays before writing the experiment metadata."""

    return {
        key: value
        for key, value in result.items()
        if key not in {"validation_indices", "prediction"}
    }


def main() -> None:
    # 限制 CPU 线程，避免特征准备影响电脑响应。
    # Limit CPU threads so feature preparation keeps the computer responsive.
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    fold_dir = RUN_DIR / FOLD_NAME
    fold_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_feather(
        PROJECT_DIR / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    base, base_columns = gru.load_relative319(PROJECT_DIR, labels)
    months = labels["month"].to_numpy(dtype=np.int16, copy=True)
    target = labels["target"].to_numpy(dtype=np.float32, copy=True)
    sample_ids = labels["sample_id"].to_numpy(copy=True)
    xs40, xs_columns = recipe.add_relative_features(base, months, base_columns)

    order_frame = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_order_quote_position_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    market_frame = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_market_microstructure_features.feather",
        columns=["sample_id", *MARKET_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(sample_ids, order_frame["sample_id"].to_numpy()):
        raise AssertionError("Order-position sample IDs do not match labels.")
    if not np.array_equal(sample_ids, market_frame["sample_id"].to_numpy()):
        raise AssertionError("Market-trajectory sample IDs do not match labels.")

    order_values = order_frame[ORDER_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    market_values = market_frame[MARKET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    features = np.concatenate([base, xs40, order_values, market_values], axis=1)
    if features.shape[1] != 385:
        raise AssertionError(f"Expected 385 features, got {features.shape[1]}.")
    if len(set([*base_columns, *xs_columns, *ORDER_COLUMNS, *MARKET_COLUMNS])) != 385:
        raise AssertionError("Expected 385 unique feature names.")
    del labels, order_frame, market_frame, order_values, market_values, base, xs40
    gc.collect()

    def make_deepwide_model(input_dimension: int, model_device: torch.device):
        # 保留官方 TabM 的成员化结构，只采用公开方案的更深更宽容量。
        # Keep official TabM members and adopt the public recipe's larger capacity.
        return recipe.TabM.make(
            n_num_features=input_dimension,
            cat_cardinalities=None,
            d_out=1,
            k=32,
            n_blocks=3,
            d_block=512,
            dropout=0.15,
            arch_type="tabm",
        ).to(model_device)

    recipe.make_model = make_deepwide_model
    candidate = recipe.run_model(
        f"{FOLD_NAME}_relative319_xs40_order20_market6_deepwide",
        features,
        target,
        months,
        59,
        62,
        70,
        torch.device("cuda"),
    )

    baseline_dir = (
        PROJECT_DIR
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-032-MARKET385-K32"
        / FOLD_NAME
    )
    baseline = json.loads(
        (baseline_dir / "candidate_result.json").read_text(encoding="utf-8")
    )
    baseline_predictions = pd.read_feather(
        baseline_dir / "validation_predictions.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    validation_indices = candidate["validation_indices"]
    if not np.array_equal(
        sample_ids[validation_indices], baseline_predictions["sample_id"].to_numpy()
    ):
        raise AssertionError("Baseline and candidate validation rows differ.")

    candidate_score = float(candidate["metrics"][SCORE_KEY])
    baseline_score = float(baseline["metrics"][SCORE_KEY])
    prediction_frame = pd.DataFrame(
        {
            "sample_id": sample_ids[validation_indices],
            "month": months[validation_indices],
            "target": target[validation_indices],
            "baseline_k32_d256_b2": baseline_predictions["candidate"].to_numpy(
                dtype=np.float64
            ),
            "candidate_k32_d512_b3": candidate["prediction"],
        }
    )
    prediction_frame.to_feather(fold_dir / "validation_predictions.feather")
    (fold_dir / "candidate_result.json").write_text(
        json.dumps(compact_result(candidate), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "validation": "train 0-59; purge 60-61; validate 62-70 excluding 66",
        "features": "Relative319 + XS40 + order20 + market6",
        "baseline": {
            "k": 32,
            "n_blocks": 2,
            "d_block": 256,
            "dropout": 0.10,
            "score": baseline_score,
        },
        "candidate": {
            "k": 32,
            "n_blocks": 3,
            "d_block": 512,
            "dropout": 0.15,
            "score": candidate_score,
        },
        "delta": candidate_score - baseline_score,
        "decision": "promote" if candidate_score > baseline_score else "reject",
    }
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
