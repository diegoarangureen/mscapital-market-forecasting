"""Append 8 target-free liquidity shock-recovery features to TabM385-k32."""

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
FEATURE_COLUMNS = "mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 realized_volatility_60 mid_momentum_60".split()
RECOVERY_COLUMNS = [
    "spread_shock_count_60",
    "spread_recovery_fraction_6",
    "spread_recovery_fraction_15",
    "depth_shock_count_60",
    "depth_recovery_fraction_6",
    "depth_recovery_fraction_15",
    "price_reversion_fraction_15",
    "liquidity_recovery_coverage_60",
]


EXPERIMENT_ID = "EXP-TABM-039-LIQUIDITY-RECOVERY8"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
FOLDS = {
    "train059_valid6270_ex66": (59, 62, 70, "months_62_70_without_66"),
}


def save_model_result(result: dict, path: Path) -> None:
    """Save the compact part of one model result without NumPy arrays."""

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
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_feather(
        PROJECT_DIR / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    base, base_columns = gru.load_relative319(PROJECT_DIR, labels)
    months = labels["month"].to_numpy(dtype=np.int16, copy=True)
    target = labels["target"].to_numpy(dtype=np.float32, copy=True)
    sample_ids = labels["sample_id"].to_numpy(copy=True)

    # XS40 remains frozen so the only experimental change is the new 6 columns.
    # 固定原有 XS40，确保实验唯一变量是新增的6列。
    xs40, xs_columns = recipe.add_relative_features(base, months, base_columns)
    state = pd.read_feather(
        PROJECT_DIR
        / "data"
        / "processed"
        / "train_market_microstructure_features.feather",
        columns=["sample_id", *FEATURE_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    if list(state.columns) != ["sample_id", *FEATURE_COLUMNS]:
        raise AssertionError("Unexpected market-trajectory feature columns.")
    if not np.array_equal(sample_ids, state["sample_id"].to_numpy()):
        raise AssertionError("Market-trajectory sample IDs do not match labels.")
    state_values = state[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    order_frame = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_order_quote_position_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    if list(order_frame.columns) != ["sample_id", *ORDER_COLUMNS]:
        raise AssertionError("Unexpected order-position feature columns.")
    if not np.array_equal(sample_ids, order_frame["sample_id"].to_numpy()):
        raise AssertionError("Order-position sample IDs do not match labels.")
    order_values = order_frame[ORDER_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    del state, order_frame, labels
    gc.collect()

    recovery_frame = pd.read_feather(
        PROJECT_DIR / "data/processed/train_liquidity_recovery_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    if list(recovery_frame.columns) != ["sample_id", *RECOVERY_COLUMNS]:
        raise AssertionError("Unexpected liquidity-recovery columns.")
    if not np.array_equal(sample_ids, recovery_frame["sample_id"].to_numpy()):
        raise AssertionError("Liquidity-recovery sample IDs do not align.")
    recovery_values = recovery_frame[RECOVERY_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    del recovery_frame

    if len(set([*base_columns, *xs_columns, *ORDER_COLUMNS, *FEATURE_COLUMNS, *RECOVERY_COLUMNS])) != 393:
        raise AssertionError("Expected 393 unique candidate columns.")

    all_results: dict[str, dict] = {}
    score_rows: list[dict] = []
    device = torch.device("cuda")

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

    recipe.make_model = make_model_k32
    for fold_name, (train_end, valid_start, valid_end, score_key) in FOLDS.items():
        fold_dir = RUN_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Reuse the identical seed42 order20 result from EXP026.
        baseline_dir = (
            PROJECT_DIR
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-032-MARKET385-K32"
            / fold_name
        )
        baseline = json.loads(
            (baseline_dir / "candidate_result.json").read_text(encoding="utf-8")
        )
        baseline_predictions = pd.read_feather(
            baseline_dir / "validation_predictions.feather"
        ).sort_values("sample_id").reset_index(drop=True)
        validation_indices = np.flatnonzero(
            (months >= valid_start) & (months <= valid_end)
        )
        if not np.array_equal(
            sample_ids[validation_indices],
            baseline_predictions["sample_id"].to_numpy(),
        ):
            raise AssertionError("Reused baseline rows do not match this fold.")
        baseline["validation_indices"] = validation_indices
        baseline["prediction"] = baseline_predictions["candidate"].to_numpy(
            dtype=np.float64, copy=True
        )
        del baseline_predictions

        candidate_features = np.concatenate([base, xs40, order_values, state_values, recovery_values], axis=1)
        if candidate_features.shape[1] != 393:
            raise AssertionError("Expected 393 candidate columns.")
        candidate = recipe.run_model(
            f"{fold_name}_tabm385_plus_recovery8",
            candidate_features,
            target,
            months,
            train_end,
            valid_start,
            valid_end,
            device,
        )
        del candidate_features
        gc.collect()

        baseline_score = float(baseline["metrics"][score_key])
        candidate_score = float(candidate["metrics"][score_key])
        delta = candidate_score - baseline_score
        validation_indices = baseline["validation_indices"]
        if not np.array_equal(
            validation_indices, candidate["validation_indices"]
        ):
            raise AssertionError("Baseline and candidate validation rows differ.")
        pd.DataFrame(
            {
                "sample_id": sample_ids[validation_indices],
                "month": months[validation_indices],
                "target": target[validation_indices],
                "baseline": baseline["prediction"],
                "candidate": candidate["prediction"],
            }
        ).to_feather(fold_dir / "validation_predictions.feather")
        save_model_result(baseline, fold_dir / "baseline_result.json")
        save_model_result(candidate, fold_dir / "candidate_result.json")
        row = {
            "fold": fold_name,
            "score_key": score_key,
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": delta,
        }
        score_rows.append(row)
        all_results[fold_name] = row
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del baseline, candidate, validation_indices
        gc.collect()
        if delta < 0.0:
            # A declining window already fails the predeclared two-window gate.
            break

    average_delta = float(np.mean([row["delta"] for row in score_rows]))
    no_window_decline = all(row["delta"] >= 0.0 for row in score_rows)
    passed = no_window_decline and average_delta >= 0.0007
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "baseline": "TabM385 k32, seed42, 15 epochs",
        "candidate": "TabM385 k32 plus 8 liquidity shock-recovery features",
        "baseline_feature_count": 385,
        "candidate_feature_count": 393,
        "single_variable_change": "append target-free liquidity recovery 8 features",
        "added_features": RECOVERY_COLUMNS,
        "folds": all_results,
        "average_delta": average_delta,
        "no_window_decline": no_window_decline,
        "stopped_early": len(score_rows) < len(FOLDS),
        "pass_threshold": 0.0007,
        "passed": passed,
        "next_action": (
            "paired_early_window" if passed else "reject_recovery_features"
        ),
    }
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
