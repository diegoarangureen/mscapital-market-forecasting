"""Evaluate the fixed blend after averaging seed-42 and seed-137 GRU signals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit
from exp_tree_071_gru_embeddings import add_window_metrics


FOLDS = {
    "train049_valid5059": (50, 59, "EXP-GRU-003-STRONG-JOINT-DEV", "EXP-GRU-005-JOINT-SEED137/train049_valid5059"),
    "train059_valid6070": (60, 70, "EXP-GRU-004-STRONG-JOINT-CONFIRM", "EXP-GRU-005-JOINT-SEED137/train059_valid6070"),
}


def load_joint(project_dir: Path, run: str) -> np.ndarray:
    frame = pd.read_feather(
        project_dir / "data" / "interim" / "sequence_experiments" / run
        / "joint_gru319" / "validation_predictions_epoch06.feather"
    )
    return frame["prediction"].to_numpy(dtype=np.float64)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-073-GRU-SEED-ENSEMBLE"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for fold_name, (valid_start, valid_end, run42, run137) in FOLDS.items():
        frame42 = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TREE-071-GRU96-EMBEDDING" / fold_name / "validation_predictions.feather"
        )
        frame137 = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TREE-072-GRU96-EMBEDDING-SEED137" / fold_name / "validation_predictions.feather"
        )
        tabm = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / fold_name / "validation_predictions.feather"
        )
        ids = frame42["sample_id"].to_numpy()
        if not np.array_equal(frame137["sample_id"].to_numpy(), ids):
            raise AssertionError("Embedding prediction IDs differ between seeds.")
        if not np.array_equal(tabm["sample_id"].to_numpy(), ids):
            raise AssertionError("TabM prediction IDs differ.")

        embedding = 0.5 * unit(frame42["gru_embedding_tree"].to_numpy(dtype=np.float64))
        embedding += 0.5 * unit(frame137["gru_embedding_tree"].to_numpy(dtype=np.float64))
        joint = 0.5 * unit(load_joint(project_dir, run42))
        joint += 0.5 * unit(load_joint(project_dir, run137))
        tree_blend = (
            0.75 * unit(tabm["tabm_trim1"].to_numpy(dtype=np.float64))
            + 0.20 * unit(frame42["relative319_tree"].to_numpy(dtype=np.float64))
            + 0.05 * unit(embedding)
        )
        prediction = 0.90 * unit(tree_blend) + 0.10 * unit(joint)
        target = frame42["target"].to_numpy(dtype=np.float64)
        months = frame42["month"].to_numpy()
        baseline = frame42["current_b001_trim1"].to_numpy(dtype=np.float64)
        metrics = evaluate(target, prediction, months, valid_start, valid_end)
        baseline_metrics = evaluate(target, baseline, months, valid_start, valid_end)
        add_window_metrics(metrics, target, prediction, months)
        add_window_metrics(baseline_metrics, target, baseline, months)
        deltas = {
            key: metrics[key] - baseline_metrics[key]
            for key in (
                "overall", "without_month_66", "months_62_70_without_66",
                "months_67_70", "monthly_std", "monthly_worst", "monthly_q25",
            )
            if key in metrics and key in baseline_metrics
        }
        output = frame42[["sample_id", "month", "target"]].copy()
        output["fixed_seed_ensemble"] = prediction
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        output.to_feather(fold_dir / "validation_predictions.feather")
        payload[fold_name] = {"metrics": metrics, "deltas_vs_current_b001": deltas}
        print(fold_name, json.dumps(deltas, ensure_ascii=False), flush=True)

    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
