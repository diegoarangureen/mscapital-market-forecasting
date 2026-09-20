"""Evaluate the preregistered 5% GRU-embedding-tree blend for EXP-TREE-072."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit
from exp_tree_071_gru_embeddings import add_window_metrics


FOLDS = {
    "train049_valid5059": (50, 59, "EXP-GRU-005-JOINT-SEED137/train049_valid5059"),
    "train059_valid6070": (60, 70, "EXP-GRU-005-JOINT-SEED137/train059_valid6070"),
}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-072-GRU96-EMBEDDING-SEED137"
    )
    results = {}
    for fold_name, (valid_start, valid_end, encoder_run) in FOLDS.items():
        fold_dir = run_dir / fold_name
        frame = pd.read_feather(fold_dir / "validation_predictions.feather")
        tabm = pd.read_feather(
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-016-RELATIVE-SCALE-ADD12"
            / fold_name
            / "validation_predictions.feather"
        )
        joint = pd.read_feather(
            project_dir
            / "data"
            / "interim"
            / "sequence_experiments"
            / encoder_run
            / "joint_gru319"
            / "validation_predictions_epoch06.feather"
        )
        expected_ids = frame["sample_id"].to_numpy()
        if not np.array_equal(tabm["sample_id"].to_numpy(), expected_ids):
            raise AssertionError("TabM validation IDs differ.")
        if not np.array_equal(joint["sample_id"].to_numpy(), expected_ids):
            raise AssertionError("Joint-GRU validation IDs differ.")

        # 总权重：67.5% TabM、18% 旧树、4.5% 嵌入树、10% Joint-GRU。
        # Total weights: 67.5% TabM, 18% old tree, 4.5% embedding tree, 10% Joint-GRU.
        tree_blend = (
            0.75 * unit(tabm["tabm_trim1"].to_numpy(dtype=np.float64))
            + 0.20 * unit(frame["relative319_tree"].to_numpy(dtype=np.float64))
            + 0.05 * unit(frame["gru_embedding_tree"].to_numpy(dtype=np.float64))
        )
        prediction = (
            0.90 * unit(tree_blend)
            + 0.10 * unit(joint["prediction"].to_numpy(dtype=np.float64))
        )
        target = frame["target"].to_numpy(dtype=np.float64)
        months = frame["month"].to_numpy()
        metrics = evaluate(target, prediction, months, valid_start, valid_end)
        add_window_metrics(metrics, target, prediction, months)
        baseline_metrics = evaluate(
            target,
            frame["current_b001_trim1"].to_numpy(dtype=np.float64),
            months,
            valid_start,
            valid_end,
        )
        add_window_metrics(
            baseline_metrics,
            target,
            frame["current_b001_trim1"].to_numpy(dtype=np.float64),
            months,
        )
        deltas = {
            key: metrics[key] - baseline_metrics[key]
            for key in (
                "overall",
                "without_month_66",
                "months_62_70_without_66",
                "months_67_70",
                "monthly_std",
                "monthly_worst",
                "monthly_q25",
            )
            if key in metrics and key in baseline_metrics
        }
        output = frame[["sample_id", "month", "target"]].copy()
        output["fixed_embedding_tree05_joint10"] = prediction
        output.to_feather(fold_dir / "fixed_weight_predictions.feather")
        results[fold_name] = {
            "weights": {
                "relative319_tabm": 0.675,
                "relative319_tree": 0.18,
                "gru_embedding_tree": 0.045,
                "joint_gru319": 0.10,
            },
            "metrics": metrics,
            "deltas_vs_current_b001": deltas,
        }
        print(fold_name, json.dumps(deltas, ensure_ascii=False), flush=True)

    (run_dir / "fixed_weight_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
