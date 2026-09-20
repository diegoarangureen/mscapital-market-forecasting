"""Confirm TabM385 PLE on the earlier train0-49 / valid50-59 window."""

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
import exp_tabm_034_market385_k32_ple as ple_experiment
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_COLUMNS


EXPERIMENT_ID = "EXP-TABM-035-MARKET385-K32-PLE32-EARLY-CONFIRM"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
FOLD_NAME = "train049_valid5059"
TRAIN_END = 49
VALID_START = 50
VALID_END = 59
FEATURE_COLUMNS = ple_experiment.FEATURE_COLUMNS


def save_result(result: dict, path: Path) -> None:
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"validation_indices", "prediction"}
    }
    path.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    recipe.SEED = 42
    RUN_DIR.mkdir(parents=True, exist_ok=True)
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
    state = pd.read_feather(
        PROJECT_DIR
        / "data"
        / "processed"
        / "train_market_microstructure_features.feather",
        columns=["sample_id", *FEATURE_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    order_frame = pd.read_feather(
        PROJECT_DIR
        / "data"
        / "processed"
        / "train_order_quote_position_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(sample_ids, state["sample_id"].to_numpy()):
        raise AssertionError("Market feature sample IDs do not match labels.")
    if not np.array_equal(sample_ids, order_frame["sample_id"].to_numpy()):
        raise AssertionError("Order feature sample IDs do not match labels.")
    features = np.concatenate(
        [
            base,
            xs40,
            order_frame[ORDER_COLUMNS].to_numpy(dtype=np.float32, copy=True),
            state[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True),
        ],
        axis=1,
    )
    del state, order_frame, labels, base, xs40
    gc.collect()
    if features.shape[1] != 385:
        raise AssertionError("Expected 385 candidate columns.")

    def make_model_k32(input_dimension: int, model_device: torch.device):
        return recipe.TabM.make(
            n_num_features=input_dimension,
            cat_cardinalities=None,
            d_out=1,
            k=32,
            n_blocks=2,
            d_block=256,
            dropout=0.1,
            arch_type="tabm",
        ).to(model_device)

    device = torch.device("cuda")
    recipe.make_model = make_model_k32
    baseline = recipe.run_model(
        f"{FOLD_NAME}_tabm385_k32",
        features,
        target,
        months,
        TRAIN_END,
        VALID_START,
        VALID_END,
        device,
    )
    candidate = ple_experiment.run_model_ple(
        f"{FOLD_NAME}_tabm385_k32_ple32",
        features,
        target,
        months,
        TRAIN_END,
        VALID_START,
        VALID_END,
        device,
    )
    if not np.array_equal(
        baseline["validation_indices"], candidate["validation_indices"]
    ):
        raise AssertionError("Baseline and PLE validation rows differ.")
    validation_indices = baseline["validation_indices"]
    baseline_score = float(baseline["metrics"]["overall"])
    candidate_score = float(candidate["metrics"]["overall"])
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "fold": FOLD_NAME,
        "seed": 42,
        "baseline": baseline_score,
        "candidate": candidate_score,
        "delta": candidate_score - baseline_score,
        "late_fold_delta": 0.00015822294631220868,
        "both_windows_positive": candidate_score >= baseline_score,
        "next_action": (
            "seed137_late_confirmation"
            if candidate_score >= baseline_score
            else "reject_ple_for_fulltrain"
        ),
    }
    pd.DataFrame(
        {
            "sample_id": sample_ids[validation_indices],
            "month": months[validation_indices],
            "target": target[validation_indices],
            "baseline": baseline["prediction"],
            "candidate": candidate["prediction"],
        }
    ).to_feather(fold_dir / "validation_predictions.feather")
    save_result(baseline, fold_dir / "baseline_result.json")
    save_result(candidate, fold_dir / "candidate_result.json")
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
