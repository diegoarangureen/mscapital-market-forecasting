"""Test a fixed equal split of the promoted 10% sequence-model allocation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit
from exp_tree_071_gru_embeddings import add_window_metrics


FOLDS = {
    "train049_valid5059": {
        "valid": (50, 59),
        "transformer": "EXP-TRANSFORMER-001-JOINT-DEV",
        "joint42": "EXP-GRU-003-STRONG-JOINT-DEV",
        "joint137": "EXP-GRU-005-JOINT-SEED137/train049_valid5059",
    },
    "train059_valid6070": {
        "valid": (60, 70),
        "transformer": "EXP-TRANSFORMER-002-JOINT-CONFIRM",
        "joint42": "EXP-GRU-004-STRONG-JOINT-CONFIRM",
        "joint137": "EXP-GRU-005-JOINT-SEED137/train059_valid6070",
    },
}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def load_sequence_prediction(project_dir: Path, run: str) -> pd.DataFrame:
    path = project_dir / "data" / "interim" / "sequence_experiments" / run
    if run.startswith("EXP-TRANSFORMER"):
        return pd.read_feather(path / "validation_predictions_epoch06.feather")
    return pd.read_feather(path / "joint_gru319" / "validation_predictions_epoch06.feather")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = (
        project_dir / "data" / "interim" / "sequence_experiments"
        / "EXP-TRANSFORMER-003-GRU-BLEND"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for fold_name, config in FOLDS.items():
        valid_start, valid_end = config["valid"]
        transformer_frame = load_sequence_prediction(project_dir, config["transformer"])
        joint42_frame = load_sequence_prediction(project_dir, config["joint42"])
        joint137_frame = load_sequence_prediction(project_dir, config["joint137"])
        tree42 = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TREE-071-GRU96-EMBEDDING" / fold_name / "validation_predictions.feather"
        )
        tree137 = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TREE-072-GRU96-EMBEDDING-SEED137" / fold_name
            / "validation_predictions.feather"
        )
        tabm = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / fold_name
            / "validation_predictions.feather"
        )
        ids = transformer_frame["sample_id"].to_numpy()
        for frame in (joint42_frame, joint137_frame, tree42, tree137, tabm):
            if not np.array_equal(frame["sample_id"].to_numpy(), ids):
                raise AssertionError(f"Prediction IDs differ for {fold_name}.")

        baseline = transformer_frame["b001_trim1"].to_numpy(dtype=np.float64)
        transformer = transformer_frame["prediction"].to_numpy(dtype=np.float64)
        joint = 0.5 * unit(joint42_frame["prediction"].to_numpy(dtype=np.float64))
        joint += 0.5 * unit(joint137_frame["prediction"].to_numpy(dtype=np.float64))
        embedding = 0.5 * unit(tree42["gru_embedding_tree"].to_numpy(dtype=np.float64))
        embedding += 0.5 * unit(tree137["gru_embedding_tree"].to_numpy(dtype=np.float64))
        tree_basket = (
            0.75 * unit(tabm["tabm_trim1"].to_numpy(dtype=np.float64))
            + 0.20 * unit(tree42["relative319_tree"].to_numpy(dtype=np.float64))
            + 0.05 * unit(embedding)
        )
        predictions = {
            "current_b001": baseline,
            "b00190_transformer10": 0.90 * unit(baseline) + 0.10 * unit(transformer),
            "b00190_joint10": 0.90 * unit(baseline) + 0.10 * unit(joint),
            "promoted_tree90_joint10": 0.90 * unit(tree_basket) + 0.10 * unit(joint),
            "tree90_joint05_transformer05": (
                0.90 * unit(tree_basket)
                + 0.05 * unit(joint)
                + 0.05 * unit(transformer)
            ),
        }
        target = transformer_frame["target"].to_numpy(dtype=np.float64)
        months = transformer_frame["month"].to_numpy()
        metrics = {}
        for name, prediction in predictions.items():
            values = evaluate(target, prediction, months, valid_start, valid_end)
            add_window_metrics(values, target, prediction, months)
            metrics[name] = values
        baseline_metrics = metrics["current_b001"]
        deltas = {}
        for name, values in metrics.items():
            deltas[name] = {
                key: values[key] - baseline_metrics[key]
                for key in (
                    "overall", "without_month_66", "months_62_70_without_66",
                    "months_67_70", "monthly_std", "monthly_worst", "monthly_q25",
                )
                if key in values and key in baseline_metrics
            }
        correlations = {
            "transformer_vs_b001_cosine": cosine(transformer, baseline),
            "transformer_vs_joint_cosine": cosine(transformer, joint),
        }
        result[fold_name] = {
            "metrics": metrics,
            "deltas_vs_current_b001": deltas,
            "prediction_geometry": correlations,
        }
        print(fold_name, json.dumps({"deltas": deltas, **correlations}, ensure_ascii=False), flush=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
