"""Evaluate the preregistered 4.5% effective LightGBM blend."""

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
        "lightgbm": "data/interim/tree_experiments/EXP-TREE-074-LIGHTGBM800-TRAIN049-VALID5059/validation_predictions.feather",
        "joint42": "EXP-GRU-003-STRONG-JOINT-DEV",
        "joint137": "EXP-GRU-005-JOINT-SEED137/train049_valid5059",
    },
    "train059_valid6070": {
        "valid": (60, 70),
        "lightgbm": "outputs/predictions/exp-tree-067_valid.feather",
        "joint42": "EXP-GRU-004-STRONG-JOINT-CONFIRM",
        "joint137": "EXP-GRU-005-JOINT-SEED137/train059_valid6070",
    },
}


def load_joint(project_dir: Path, run: str) -> pd.DataFrame:
    return pd.read_feather(
        project_dir / "data" / "interim" / "sequence_experiments" / run
        / "joint_gru319" / "validation_predictions_epoch06.feather"
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = (
        project_dir / "data" / "interim" / "tree_experiments"
        / "EXP-TREE-075-LIGHTGBM-FIXED-BLEND"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for fold_name, config in FOLDS.items():
        valid_start, valid_end = config["valid"]
        tabm = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / fold_name
            / "validation_predictions.feather"
        )
        tree42 = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TREE-071-GRU96-EMBEDDING" / fold_name
            / "validation_predictions.feather"
        )
        tree137 = pd.read_feather(
            project_dir / "data" / "interim" / "tree_experiments"
            / "EXP-TREE-072-GRU96-EMBEDDING-SEED137" / fold_name
            / "validation_predictions.feather"
        )
        lightgbm = pd.read_feather(project_dir / config["lightgbm"])
        joint42 = load_joint(project_dir, config["joint42"])
        joint137 = load_joint(project_dir, config["joint137"])
        ids = tabm["sample_id"].to_numpy()
        for frame in (tree42, tree137, lightgbm, joint42, joint137):
            if not np.array_equal(frame["sample_id"].to_numpy(), ids):
                raise AssertionError(f"Prediction IDs differ for {fold_name}.")

        tabm_prediction = tabm["tabm_trim1"].to_numpy(dtype=np.float64)
        old_xgboost = tabm["xgboost"].to_numpy(dtype=np.float64)
        embedding = 0.5 * unit(tree42["gru_embedding_tree"].to_numpy(dtype=np.float64))
        embedding += 0.5 * unit(tree137["gru_embedding_tree"].to_numpy(dtype=np.float64))
        joint = 0.5 * unit(joint42["prediction"].to_numpy(dtype=np.float64))
        joint += 0.5 * unit(joint137["prediction"].to_numpy(dtype=np.float64))
        lightgbm_prediction = lightgbm["prediction"].to_numpy(dtype=np.float64)

        current_tree_basket = (
            0.75 * unit(tabm_prediction)
            + 0.20 * unit(old_xgboost)
            + 0.05 * unit(embedding)
        )
        lightgbm_tree_basket = (
            0.75 * unit(tabm_prediction)
            + 0.15 * unit(old_xgboost)
            + 0.05 * unit(lightgbm_prediction)
            + 0.05 * unit(embedding)
        )
        predictions = {
            "public_best_local_recipe": 0.90 * unit(current_tree_basket) + 0.10 * unit(joint),
            "lightgbm_effective045": 0.90 * unit(lightgbm_tree_basket) + 0.10 * unit(joint),
        }
        target = tabm["target"].to_numpy(dtype=np.float64)
        months = tabm["month"].to_numpy()
        metrics = {}
        for name, prediction in predictions.items():
            values = evaluate(target, prediction, months, valid_start, valid_end)
            add_window_metrics(values, target, prediction, months)
            metrics[name] = values
        current_metrics = metrics["public_best_local_recipe"]
        deltas = {
            key: metrics["lightgbm_effective045"][key] - current_metrics[key]
            for key in (
                "overall", "without_month_66", "months_62_70_without_66",
                "months_67_70", "monthly_std", "monthly_worst", "monthly_q25",
            )
            if key in current_metrics
        }
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "sample_id": ids,
                "month": months,
                "target": target,
                **predictions,
            }
        ).to_feather(fold_dir / "validation_predictions.feather")
        result[fold_name] = {
            "weights": {
                "current_effective": {
                    "tabm": 0.675, "old_xgboost": 0.18,
                    "gru_embedding_tree": 0.045, "joint_gru": 0.10,
                },
                "candidate_effective": {
                    "tabm": 0.675, "old_xgboost": 0.135,
                    "lightgbm": 0.045, "gru_embedding_tree": 0.045,
                    "joint_gru": 0.10,
                },
            },
            "metrics": metrics,
            "candidate_deltas_vs_public_best_local_recipe": deltas,
        }
        print(fold_name, json.dumps(deltas, ensure_ascii=False), flush=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
