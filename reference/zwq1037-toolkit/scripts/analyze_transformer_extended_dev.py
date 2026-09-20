"""Compare continued Transformer checkpoints against the exact public-best recipe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit


EPOCHS = (6, 8, 10, 12)


def load_joint(project_dir: Path, run: str) -> pd.DataFrame:
    return pd.read_feather(
        project_dir / "data" / "interim" / "sequence_experiments" / run
        / "joint_gru319" / "validation_predictions_epoch06.feather"
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    fold_name = "train049_valid5059"
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
    joint42 = load_joint(project_dir, "EXP-GRU-003-STRONG-JOINT-DEV")
    joint137 = load_joint(
        project_dir, "EXP-GRU-005-JOINT-SEED137/train049_valid5059"
    )
    ids = tabm["sample_id"].to_numpy()
    for frame in (tree42, tree137, joint42, joint137):
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError("Validation prediction IDs differ.")
    embedding = 0.5 * unit(tree42["gru_embedding_tree"].to_numpy(dtype=np.float64))
    embedding += 0.5 * unit(tree137["gru_embedding_tree"].to_numpy(dtype=np.float64))
    joint = 0.5 * unit(joint42["prediction"].to_numpy(dtype=np.float64))
    joint += 0.5 * unit(joint137["prediction"].to_numpy(dtype=np.float64))
    tree_basket = (
        0.75 * unit(tabm["tabm_trim1"].to_numpy(dtype=np.float64))
        + 0.20 * unit(tabm["xgboost"].to_numpy(dtype=np.float64))
        + 0.05 * unit(embedding)
    )
    current = 0.90 * unit(tree_basket) + 0.10 * unit(joint)
    target = tabm["target"].to_numpy(dtype=np.float64)
    months = tabm["month"].to_numpy()
    current_metrics = evaluate(target, current, months, 50, 59)
    rows = []
    output = tabm[["sample_id", "month", "target"]].copy()
    output["public_best_local_recipe"] = current
    transformer_dir = (
        project_dir / "data" / "interim" / "sequence_experiments"
        / "EXP-TRANSFORMER-001-JOINT-DEV"
    )
    for epoch in EPOCHS:
        frame = pd.read_feather(
            transformer_dir / f"validation_predictions_epoch{epoch:02d}.feather"
        )
        if not np.array_equal(frame["sample_id"].to_numpy(), ids):
            raise AssertionError(f"Transformer epoch {epoch} IDs differ.")
        transformer_prediction = frame["prediction"].to_numpy(dtype=np.float64)
        candidate = (
            0.90 * unit(tree_basket)
            + 0.05 * unit(joint)
            + 0.05 * unit(transformer_prediction)
        )
        metrics = evaluate(target, candidate, months, 50, 59)
        rows.append(
            {
                "epoch": epoch,
                "standalone_cosine": evaluate(
                    target, transformer_prediction, months, 50, 59
                )["overall"],
                "full_blend_cosine": metrics["overall"],
                "full_blend_delta": metrics["overall"] - current_metrics["overall"],
                "monthly_std_delta": metrics["monthly_std"] - current_metrics["monthly_std"],
                "monthly_worst_delta": metrics["monthly_worst"] - current_metrics["monthly_worst"],
                "monthly_q25_delta": metrics["monthly_q25"] - current_metrics["monthly_q25"],
            }
        )
        output[f"epoch{epoch:02d}_candidate"] = candidate
    result_dir = (
        project_dir / "data" / "interim" / "sequence_experiments"
        / "EXP-TRANSFORMER-005-EXTENDED-DEV"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_dir / "summary.csv", index=False)
    output.to_feather(result_dir / "validation_predictions.feather")
    result = {
        "status": "complete",
        "comparison": "exact local reconstruction of Public 0.135 recipe",
        "current_metrics": current_metrics,
        "rows": rows,
        "promotion_threshold": 0.0003,
    }
    (result_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
