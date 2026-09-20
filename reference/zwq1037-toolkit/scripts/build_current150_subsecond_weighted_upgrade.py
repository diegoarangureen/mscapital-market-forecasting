"""Build a chosen subsecond-event replacement of the Transformer blend slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT / "outputs" / "submissions"
METADATA = PROJECT / "outputs" / "submission_metadata"
TRANSFORMER_SLOT_WEIGHT = 0.16
COMMON_GROUPS = 38


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction))
    )


def load_submission(path: Path, expected_ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if expected_ids is not None and not np.array_equal(
        frame["sample_id"].to_numpy(), expected_ids
    ):
        raise AssertionError(f"IDs do not align: {path}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Nonfinite predictions: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-weight", type=float, required=True)
    args = parser.parse_args()
    event_weight = args.event_weight
    if not 0.0 <= event_weight <= 1.0:
        raise ValueError("--event-weight must be in [0, 1]")
    old_weight = 1.0 - event_weight
    weight_tag = int(round(100 * event_weight))
    run_name = f"current150_subsecond_transformer{weight_tag:02d}_upgrade"

    validation_old_path = (
        PROJECT
        / "outputs/kaggle_transformer_v7_result/factorized_transformer"
        / "validation_predictions.csv"
    )
    validation_event_path = (
        PROJECT
        / "data/interim/tree_experiments"
        / "EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER"
        / "validation_predictions.feather"
    )
    old_validation = pd.read_csv(validation_old_path).rename(
        columns={"prediction": "old"}
    )
    event_validation = pd.read_feather(validation_event_path).rename(
        columns={"prediction": "event"}
    )
    validation = old_validation[["sample_id", "month", "target", "old"]].merge(
        event_validation[["sample_id", "event"]],
        on="sample_id",
        validate="one_to_one",
    )
    selection = validation["month"].isin([62, 63, 64, 65])
    forward = validation["month"].isin([67, 68, 69, 70])
    all_late = selection | forward
    old_rms = float(np.sqrt(np.mean(validation.loc[selection, "old"] ** 2)))
    event_rms = float(np.sqrt(np.mean(validation.loc[selection, "event"] ** 2)))
    validation["hybrid"] = (
        old_weight * validation["old"]
        + event_weight * validation["event"] * old_rms / event_rms
    )

    evidence = {}
    for name, mask in {
        "selection_62_65": selection,
        "forward_67_70": forward,
        "all_62_70_without_66": all_late,
    }.items():
        baseline_score = cosine(
            validation.loc[mask, "target"].to_numpy(),
            validation.loc[mask, "old"].to_numpy(),
        )
        candidate_score = cosine(
            validation.loc[mask, "target"].to_numpy(),
            validation.loc[mask, "hybrid"].to_numpy(),
        )
        evidence[name] = {
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": candidate_score - baseline_score,
        }

    paths = {
        "base": SUBMISSIONS
        / "factorized_full_replace_corr095_current138k32_equal38_common.csv",
        "old_transformer": PROJECT
        / "data/interim/kaggle_outputs/multistream_factorized_transformer_fulltrain"
        / "factorized_transformer/factorized_transformer379_full_e4.csv",
        "event_model": SUBMISSIONS
        / "standalone_subsecond_event_transformer379_full_e3.csv",
    }
    base = load_submission(paths["base"])
    ids = base["sample_id"].to_numpy()
    old = load_submission(paths["old_transformer"], ids)["prediction"].to_numpy(
        dtype=np.float64
    )
    event = load_submission(paths["event_model"], ids)["prediction"].to_numpy(
        dtype=np.float64
    )
    hybrid = old_weight * old + event_weight * event * old_rms / event_rms

    delta = TRANSFORMER_SLOT_WEIGHT * (unit(hybrid) - unit(old))
    groups = (np.arange(len(ids), dtype=np.int64) * COMMON_GROUPS // len(ids)).astype(
        np.int16
    )
    for group in np.unique(groups):
        mask = groups == group
        delta[mask] -= delta[mask].mean()

    base_prediction = base["prediction"].to_numpy(dtype=np.float64)
    prediction = base_prediction + delta
    output_path = SUBMISSIONS / f"{run_name}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
        output_path, index=False
    )
    report = {
        "run_name": run_name,
        "base_public_lb": 0.150,
        "event_weight_inside_transformer_slot": event_weight,
        "transformer_slot_weight": TRANSFORMER_SLOT_WEIGHT,
        "effective_event_weight_in_full_blend": event_weight
        * TRANSFORMER_SLOT_WEIGHT,
        "validation_evidence": evidence,
        "paths": {name: str(path) for name, path in paths.items()},
        "test_diagnostics": {
            "old_vs_hybrid_correlation": float(np.corrcoef(old, hybrid)[0, 1]),
            "candidate_vs_base_correlation": float(
                np.corrcoef(base_prediction, prediction)[0, 1]
            ),
            "rows": len(ids),
            "finite": bool(np.isfinite(prediction).all()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
        },
        "output_path": str(output_path),
        "submission_status": "prepared_not_submitted",
    }
    metadata_path = METADATA / f"{run_name}.json"
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
