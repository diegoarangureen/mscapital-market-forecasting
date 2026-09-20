"""Evaluate PLE TabM and linear-leaf LightGBM in a frozen local blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "submission_metadata"
    / "tabm_ple_linear_leaf_local_blend.json"
)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero prediction vector.")
    return vector / norm


def monthly_scores(
    target: np.ndarray, prediction: np.ndarray, months: np.ndarray
) -> dict[str, float]:
    return {
        str(int(month)): cosine(target[months == month], prediction[months == month])
        for month in np.unique(months)
    }


def main() -> None:
    experiment_dir = PROJECT_DIR / "data" / "interim" / "tree_experiments"
    tabm = pd.read_feather(
        experiment_dir
        / "EXP-TABM-032-MARKET385-K32"
        / "train059_valid6270_ex66"
        / "validation_predictions.feather"
    )
    ple = pd.read_feather(
        experiment_dir
        / "EXP-TABM-034-MARKET385-K32-PLE32"
        / "train059_valid6270_ex66"
        / "validation_predictions.feather"
    )
    tree = pd.read_feather(
        experiment_dir
        / "EXP-TREE-083-LGBM379-LINEAR-LEAF"
        / "train059_valid6270_no66"
        / "validation_predictions.feather"
    )
    transformer = pd.read_csv(
        PROJECT_DIR
        / "outputs"
        / "kaggle_transformer_v7_result"
        / "factorized_transformer"
        / "validation_predictions.csv"
    )
    real_dir = (
        experiment_dir
        / "EXP-REALMLP-009-OUR379-CORR095"
        / "train059_valid6270_ex66"
    )
    real = pd.DataFrame(
        {
            "sample_id": np.load(real_dir / "validation_sample_ids.npy"),
            "realmlp": np.load(real_dir / "validation_predictions.npy"),
            "real_target": np.load(real_dir / "validation_targets.npy"),
        }
    )

    tabm = tabm.loc[tabm["month"].ne(66)].copy()
    ple = ple.loc[ple["month"].ne(66), ["sample_id", "candidate"]].rename(
        columns={"candidate": "tabm_ple"}
    )
    tree = tree[["sample_id", "candidate_linear_leaf"]].rename(
        columns={"candidate_linear_leaf": "linear_leaf"}
    )
    transformer = transformer[["sample_id", "prediction"]].rename(
        columns={"prediction": "transformer"}
    )
    frame = (
        tabm[["sample_id", "month", "target", "candidate"]]
        .rename(columns={"candidate": "tabm_k32"})
        .merge(ple, on="sample_id", validate="one_to_one")
        .merge(tree, on="sample_id", validate="one_to_one")
        .merge(transformer, on="sample_id", validate="one_to_one")
        .merge(real, on="sample_id", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    if len(frame) != 140_806:
        raise AssertionError(f"Expected 140806 aligned rows, got {len(frame)}.")
    if not np.allclose(frame["target"], frame["real_target"], atol=2e-9, rtol=0):
        raise AssertionError("RealMLP targets are not aligned.")

    target = frame["target"].to_numpy(dtype=np.float64)
    months = frame["month"].to_numpy(dtype=np.int16)
    vectors = {
        name: frame[name].to_numpy(dtype=np.float64)
        for name in [
            "tabm_k32",
            "tabm_ple",
            "realmlp",
            "transformer",
            "linear_leaf",
        ]
    }
    normalized = {name: unit(value) for name, value in vectors.items()}
    rows: list[dict] = []

    def record(name: str, prediction: np.ndarray, details: dict) -> None:
        rows.append(
            {
                "name": name,
                "score": cosine(target, prediction),
                "monthly": monthly_scores(target, prediction, months),
                **details,
            }
        )

    record("tabm_k32", normalized["tabm_k32"], {})
    record("tabm_ple", normalized["tabm_ple"], {})
    for ple_weight in (0.1, 0.2, 0.5, 1.0):
        tabm_block = unit(
            (1.0 - ple_weight) * normalized["tabm_k32"]
            + ple_weight * normalized["tabm_ple"]
        )
        record(
            f"tabm_internal_ple_{ple_weight:.1f}",
            tabm_block,
            {"ple_weight_inside_tabm": ple_weight},
        )

        base = unit(
            0.45 * tabm_block
            + 0.17 * normalized["realmlp"]
            + 0.38 * normalized["transformer"]
        )
        record(
            f"trio_ple_{ple_weight:.1f}",
            base,
            {
                "ple_weight_inside_tabm": ple_weight,
                "base_weights": {"tabm": 0.45, "realmlp": 0.17, "transformer": 0.38},
            },
        )
        for tree_weight in (0.05, 0.10):
            blended = unit(
                (1.0 - tree_weight) * base
                + tree_weight * normalized["linear_leaf"]
            )
            record(
                f"trio_ple_{ple_weight:.1f}_linear_leaf_{tree_weight:.2f}",
                blended,
                {
                    "ple_weight_inside_tabm": ple_weight,
                    "linear_leaf_weight": tree_weight,
                },
            )

    base = unit(
        0.45 * normalized["tabm_k32"]
        + 0.17 * normalized["realmlp"]
        + 0.38 * normalized["transformer"]
    )
    record(
        "trio_k32_base",
        base,
        {"base_weights": {"tabm": 0.45, "realmlp": 0.17, "transformer": 0.38}},
    )
    for tree_weight in (0.05, 0.10):
        record(
            f"trio_k32_linear_leaf_{tree_weight:.2f}",
            unit((1.0 - tree_weight) * base + tree_weight * normalized["linear_leaf"]),
            {"linear_leaf_weight": tree_weight},
        )

    score_by_name = {row["name"]: row["score"] for row in rows}
    result = {
        "rows": len(frame),
        "months": sorted(int(month) for month in np.unique(months)),
        "fixed_blend_weights": {"tabm": 0.45, "realmlp": 0.17, "transformer": 0.38},
        "tabm_ple_single_delta": score_by_name["tabm_ple"] - score_by_name["tabm_k32"],
        "ple_10pct_trio_delta": score_by_name["trio_ple_0.1"] - score_by_name["trio_k32_base"],
        "ple_20pct_trio_delta": score_by_name["trio_ple_0.2"] - score_by_name["trio_k32_base"],
        "linear_leaf_05_trio_delta": score_by_name["trio_k32_linear_leaf_0.05"] - score_by_name["trio_k32_base"],
        "linear_leaf_10_trio_delta": score_by_name["trio_k32_linear_leaf_0.10"] - score_by_name["trio_k32_base"],
        "candidates": sorted(rows, key=lambda row: row["score"], reverse=True),
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
