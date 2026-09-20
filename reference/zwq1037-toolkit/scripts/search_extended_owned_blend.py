"""Search deployable extensions to the verified current152 owned blend."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = ROOT / "data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE"
OUT_DIR = ROOT / "data/interim/tree_experiments/EXP-BLEND-006-EXTENDED-OWNED"
REPORT = ROOT / "outputs/submission_metadata/extended_owned_blend_search_20260920.json"

SLOT_NAMES = ["old", "raw", "event", "subsecond", "multiwindow"]


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(target @ prediction / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30))


def score_bundle(target: np.ndarray, prediction: np.ndarray, months: np.ndarray) -> dict:
    selection = np.isin(months, [62, 63, 64, 65])
    forward = np.isin(months, [67, 68, 69, 70])
    return {
        "selection": cosine(target[selection], prediction[selection]),
        "forward": cosine(target[forward], prediction[forward]),
        "all": cosine(target, prediction),
        "monthly_selection": {
            str(month): cosine(target[months == month], prediction[months == month])
            for month in range(62, 66)
        },
        "monthly_forward": {
            str(month): cosine(target[months == month], prediction[months == month])
            for month in range(67, 71)
        },
    }


def rms_standardize(values: np.ndarray, selection: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64)[selection] ** 2)))
    return np.asarray(values, dtype=np.float64) / max(scale, 1e-12), scale


def simplex_weights(step: float, count: int) -> np.ndarray:
    units = int(round(1.0 / step))

    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for remainder in compositions(total - first, parts - 1):
                yield (first,) + remainder

    return np.asarray(list(compositions(units, count)), dtype=np.float64) / units


def batch_cosine(components: np.ndarray, target: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = components[mask]
    y = target[mask]
    gram = x.T @ x
    cross = x.T @ y
    numerator = weights @ cross
    denominator = np.sqrt(np.einsum("bi,ij,bj->b", weights, gram, weights) * (y @ y))
    return numerator / np.maximum(denominator, 1e-30)


def merge_prediction(frame: pd.DataFrame, source: pd.DataFrame, name: str, column: str = "prediction") -> pd.DataFrame:
    part = source[["sample_id", column]].rename(columns={column: name})
    return frame.merge(part, on="sample_id", how="inner", validate="one_to_one")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_feather(BLEND_DIR / "validation_predictions.feather")
    raw = pd.read_feather(
        ROOT / "data/interim/tree_experiments/EXP-BLEND-004-EVENT-RESIDUAL-SOFTGATE/transformer020_raw_candidate.feather"
    )
    event = pd.read_feather(
        ROOT / "data/interim/kaggle_results/market_conditioned_event_v24/market_conditioned_event_residual/validation_predictions.feather"
    )
    subsecond = pd.read_feather(
        ROOT / "data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER/validation_predictions.feather"
    )
    multiwindow = pd.read_csv(
        ROOT / "data/interim/kaggle_outputs/multistream_transformer_multiwindow_v14/factorized_transformer_multiwindow10/validation_predictions.csv"
    )
    tree = pd.read_feather(
        ROOT / "data/interim/tree_experiments/EXP-TREE-069-RELATIVE319/train059_valid6070/validation_predictions.feather"
    )
    tabm_ema = pd.read_feather(
        ROOT / "data/interim/tree_experiments/EXP-TABM-044-MARKET385-K32-EMA/train059_valid6270_ex66/validation_predictions.feather"
    )

    frame = merge_prediction(frame, raw, "raw")
    frame = merge_prediction(frame, event, "event")
    frame = merge_prediction(frame, subsecond, "subsecond")
    frame = merge_prediction(frame, multiwindow, "multiwindow")
    frame = merge_prediction(frame, tree, "tree_extra", "xgb_relative319")
    frame = merge_prediction(frame, tabm_ema, "tabm_ema", "ema")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    if len(frame) != 140_806 or not frame.sample_id.is_unique:
        raise AssertionError(f"Alignment failed: {len(frame)} rows")

    target = frame.target.to_numpy(np.float64)
    months = frame.month.to_numpy()
    selection = np.isin(months, [62, 63, 64, 65])
    forward = np.isin(months, [67, 68, 69, 70])

    core_names = ["tabm", "realmlp", "transformer", "gru"]
    core = frame[core_names].to_numpy(np.float64)
    parameters = np.load(BLEND_DIR / "fusion_parameters.npz")
    core_scale = parameters["scale"].astype(np.float64)
    standardized = core / core_scale
    volatility = frame.realized_volatility_60.to_numpy(np.float64)
    median = float(parameters["rv_median"])
    volatility = np.nan_to_num(volatility, nan=median, posinf=median, neginf=median)
    anchors = parameters["anchors"].astype(np.float64)
    state_weights = parameters["state_weights"].astype(np.float64)
    gate = np.column_stack(
        [np.interp(volatility, anchors, state_weights[:, column]) for column in range(4)]
    )
    if not np.allclose(gate.sum(axis=1), 1.0):
        raise AssertionError("Soft-gate weights do not sum to one")

    old = standardized[:, 2]
    slot_values = {"old": old}
    slot_scales = {"old": float(core_scale[2])}
    for name in SLOT_NAMES[1:]:
        slot_values[name], slot_scales[name] = rms_standardize(frame[name].to_numpy(np.float64), selection)

    base_without_slot = (standardized * gate).sum(axis=1) - gate[:, 2] * old
    slot_components = np.column_stack(
        [base_without_slot + gate[:, 2] * slot_values[name] for name in SLOT_NAMES]
    )
    slot_grid = simplex_weights(0.05, len(SLOT_NAMES))
    slot_selection_scores = batch_cosine(slot_components, target, selection, slot_grid)
    slot_index = int(np.argmax(slot_selection_scores))
    selected_slot_weights = slot_grid[slot_index]
    selected_slot_prediction = slot_components @ selected_slot_weights

    verified_joint_weights = np.asarray([0.0, 0.4, 0.6, 0.0, 0.0], dtype=np.float64)
    verified_joint_prediction = slot_components @ verified_joint_weights
    joint_metrics = score_bundle(target, verified_joint_prediction, months)
    slot_metrics = score_bundle(target, selected_slot_prediction, months)

    tree_standardized, tree_scale = rms_standardize(frame.tree_extra.to_numpy(np.float64), selection)
    tabm_ema_standardized, tabm_ema_scale = rms_standardize(frame.tabm_ema.to_numpy(np.float64), selection)

    outer_rows = []
    for tree_units in range(0, 11):
        for tabm_units in range(0, 11 - tree_units):
            tree_weight = tree_units * 0.025
            tabm_weight = tabm_units * 0.025
            base_weight = 1.0 - tree_weight - tabm_weight
            prediction = (
                base_weight * selected_slot_prediction
                + tree_weight * tree_standardized
                + tabm_weight * tabm_ema_standardized
            )
            current = score_bundle(target, prediction, months)
            outer_rows.append(
                {
                    "base_weight": base_weight,
                    "tree_weight": tree_weight,
                    "tabm_ema_weight": tabm_weight,
                    **current,
                }
            )
    selected_outer = max(outer_rows, key=lambda item: item["selection"])
    final_prediction = (
        selected_outer["base_weight"] * selected_slot_prediction
        + selected_outer["tree_weight"] * tree_standardized
        + selected_outer["tabm_ema_weight"] * tabm_ema_standardized
    )
    final_metrics = score_bundle(target, final_prediction, months)

    def delta_bundle(candidate: dict, baseline: dict) -> dict:
        return {
            "selection": candidate["selection"] - baseline["selection"],
            "forward": candidate["forward"] - baseline["forward"],
            "all": candidate["all"] - baseline["all"],
            "monthly_selection": {
                month: candidate["monthly_selection"][month] - baseline["monthly_selection"][month]
                for month in baseline["monthly_selection"]
            },
            "monthly_forward": {
                month: candidate["monthly_forward"][month] - baseline["monthly_forward"][month]
                for month in baseline["monthly_forward"]
            },
        }

    slot_delta = delta_bundle(slot_metrics, joint_metrics)
    final_delta = delta_bundle(final_metrics, joint_metrics)
    passed = bool(
        final_delta["selection"] >= 0.0003
        and final_delta["forward"] >= 0.0005
        and final_delta["all"] >= 0.0005
        and sum(value >= 0 for value in final_delta["monthly_forward"].values()) >= 3
        and min(final_delta["monthly_forward"].values()) >= -0.002
    )

    # Fixed-weight month audit; no forward month participates in selection.
    # 固定权重逐月检查；前向月份不参与权重选择。
    selection_positive_months = sum(value >= 0 for value in final_delta["monthly_selection"].values())
    forward_positive_months = sum(value >= 0 for value in final_delta["monthly_forward"].values())

    frame["verified_joint"] = verified_joint_prediction
    frame["selected_extended"] = final_prediction
    frame.to_feather(OUT_DIR / "validation_predictions.feather")
    pd.DataFrame(outer_rows).sort_values("selection", ascending=False).to_csv(
        OUT_DIR / "outer_grid.csv", index=False
    )

    top_slot = np.argsort(slot_selection_scores)[-10:][::-1]
    report = {
        "experiment": "EXP-BLEND-006-EXTENDED-OWNED",
        "status": "complete",
        "selection_rule": "Optimize only months62-65; months67-70 are a fixed gate.",
        "models_considered": {
            "transformer_slot": SLOT_NAMES,
            "outer_owned": ["tree", "tabm_ema"],
        },
        "candidate_files_available": True,
        "slot_grid_size": int(len(slot_grid)),
        "slot_scales": slot_scales,
        "selected_slot_weights": dict(zip(SLOT_NAMES, selected_slot_weights.tolist())),
        "selected_slot_metrics": slot_metrics,
        "selected_slot_delta_vs_verified_joint": slot_delta,
        "top10_slot_by_selection": [
            {
                "weights": dict(zip(SLOT_NAMES, slot_grid[index].tolist())),
                "selection": float(slot_selection_scores[index]),
            }
            for index in top_slot
        ],
        "outer_grid_size": len(outer_rows),
        "selected_outer_weights": {
            "selected_transformer_softgate": selected_outer["base_weight"],
            "tree": selected_outer["tree_weight"],
            "tabm_ema": selected_outer["tabm_ema_weight"],
        },
        "outer_scales": {"tree": tree_scale, "tabm_ema": tabm_ema_scale},
        "verified_joint_metrics": joint_metrics,
        "final_metrics": final_metrics,
        "final_delta_vs_verified_joint": final_delta,
        "selection_positive_months": selection_positive_months,
        "forward_positive_months": forward_positive_months,
        "gate": {
            "selection_delta_min": 0.0003,
            "forward_delta_min": 0.0005,
            "all_delta_min": 0.0005,
            "forward_months_nonnegative_min": 3,
            "worst_forward_month_delta_min": -0.002,
        },
        "passed": passed,
        "formal_submission": False,
        "limitation": "Public block has no labels offline. Search changes only the owned block and transfers its selected weights to full-test predictions.",
    }
    (OUT_DIR / "score_only.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "selected_slot_weights": report["selected_slot_weights"],
                "selected_outer_weights": report["selected_outer_weights"],
                "final_delta_vs_verified_joint": final_delta,
                "selection_positive_months": selection_positive_months,
                "forward_positive_months": forward_positive_months,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
