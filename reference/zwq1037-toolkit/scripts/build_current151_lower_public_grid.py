"""Reduce redundant public-model weight in the scored 0.151 blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factorized_transformer_lb_candidates import build_common_component


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "outputs" / "submissions"
METADATA_PATH = (
    PROJECT / "outputs/submission_metadata/current151_lower_public_grid.json"
)
EXPECTED_ROWS = 647_896

PATHS = {
    "gpu": PROJECT
    / "data/external/kaggle_public/bestwater_cos689/output/submission.csv",
    "tpu": PROJECT
    / "data/external/kaggle_public/bestwater_tpu_v6/output/submission.csv",
    "yangq": PROJECT / "data/external/kaggle_public/yangq_lb142/output/submission.csv",
    "tabm": PROJECT / "outputs/submissions/tabm385_k32_temporal3fold_3seed.csv",
    "realmlp": PROJECT
    / "outputs/submissions/realmlp379_rq16_temporal3fold_e8_fold_end57.csv",
    "transformer_old": PROJECT
    / "outputs/submissions/standalone_factorized_transformer379.csv",
    "transformer_multi": PROJECT
    / "data/interim/kaggle_outputs/multistream_transformer_multiwindow_v15_full"
    / "factorized_transformer/factorized_transformer379_multiwindow10_full_e3.csv",
    "gru": PROJECT
    / "outputs/submissions/standalone_timeaware_three_stream_gru379_temporal3fold.csv",
}

INCUMBENT_WEIGHTS = {
    "gpu": 0.20,
    "tpu": 0.12,
    "yangq": 0.32,
    "tabm": 0.12,
    "realmlp": 0.03,
    "transformer_old": 0.09,
    "transformer_multi": 0.09,
    "gru": 0.03,
}

CANDIDATES = {
    "public60": {
        "gpu": 0.20,
        "tpu": 0.08,
        "yangq": 0.32,
        "tabm": 0.13,
        "realmlp": 0.03,
        "transformer_old": 0.095,
        "transformer_multi": 0.095,
        "gru": 0.05,
    },
    "public56": {
        "gpu": 0.20,
        "tpu": 0.06,
        "yangq": 0.30,
        "tabm": 0.14,
        "realmlp": 0.03,
        "transformer_old": 0.10,
        "transformer_multi": 0.10,
        "gru": 0.07,
    },
    "public56_tpu4": {
        "gpu": 0.20,
        "tpu": 0.04,
        "yangq": 0.32,
        "tabm": 0.14,
        "realmlp": 0.03,
        "transformer_old": 0.10,
        "transformer_multi": 0.10,
        "gru": 0.07,
    },
    "public52_tpu0": {
        "gpu": 0.20,
        "tpu": 0.00,
        "yangq": 0.32,
        "tabm": 0.15,
        "realmlp": 0.03,
        "transformer_old": 0.105,
        "transformer_multi": 0.105,
        "gru": 0.09,
    },    "public52": {
        "gpu": 0.19,
        "tpu": 0.04,
        "yangq": 0.29,
        "tabm": 0.15,
        "realmlp": 0.03,
        "transformer_old": 0.105,
        "transformer_multi": 0.105,
        "gru": 0.09,
    },
    "public30": {
        "gpu": 0.10,
        "tpu": 0.02,
        "yangq": 0.18,
        "tabm": 0.22,
        "realmlp": 0.03,
        "transformer_old": 0.15,
        "transformer_multi": 0.15,
        "gru": 0.15,
    },
    "public48": {
        "gpu": 0.18,
        "tpu": 0.02,
        "yangq": 0.28,
        "tabm": 0.16,
        "realmlp": 0.03,
        "transformer_old": 0.11,
        "transformer_multi": 0.11,
        "gru": 0.11,
    },
}


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if len(frame) != EXPECTED_ROWS or not frame["sample_id"].is_unique:
        raise AssertionError(f"Invalid rows: {path}")
    if not np.array_equal(frame["sample_id"].to_numpy(), expected_ids):
        raise AssertionError(f"IDs differ: {path}")
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError(f"Nonfinite prediction: {path}")
    return prediction


def add_common_component(
    prediction: np.ndarray,
    groups: np.ndarray,
    common: dict[int, float],
    target_scale: float,
) -> np.ndarray:
    output = prediction.copy()
    for group in np.unique(groups):
        mask = groups == group
        output[mask] = (
            output[mask] - output[mask].mean() + common[int(group)] / target_scale
        )
    return output


def build(
    weights: dict[str, float],
    normalized: dict[str, np.ndarray],
    groups: np.ndarray,
    common: dict[int, float],
    target_scale: float,
) -> np.ndarray:
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise AssertionError(f"Weights do not sum to one: {weights}")
    before_common = sum(weights[name] * normalized[name] for name in weights)
    return add_common_component(before_common, groups, common, target_scale)


def main() -> None:
    labels, template, groups, common = build_common_component()
    ids = template["sample_id"].to_numpy()
    source = {name: load(path, ids) for name, path in PATHS.items()}
    normalized = {name: unit(values) for name, values in source.items()}
    target_scale = float(labels["target"].to_numpy(dtype=np.float64).std())

    reconstructed = build(
        INCUMBENT_WEIGHTS, normalized, groups, common, target_scale
    )
    incumbent_path = OUTPUT_DIR / "current151_structure_transformer50.csv"
    incumbent = load(incumbent_path, ids)
    reconstruction = {
        "correlation": float(np.corrcoef(reconstructed, incumbent)[0, 1]),
        "max_abs_error": float(np.max(np.abs(reconstructed - incumbent))),
        "rms_error": float(np.sqrt(np.mean((reconstructed - incumbent) ** 2))),
    }
    if reconstruction["max_abs_error"] > 1e-10:
        raise AssertionError(f"Failed to reconstruct incumbent: {reconstruction}")

    outputs = []
    for label, weights in CANDIDATES.items():
        prediction = build(weights, normalized, groups, common, target_scale)
        run_name = f"current151_lower_{label}"
        output_path = OUTPUT_DIR / f"{run_name}.csv"
        pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
            output_path, index=False
        )
        public_weight = weights["gpu"] + weights["tpu"] + weights["yangq"]
        outputs.append(
            {
                "run_name": run_name,
                "weights": weights,
                "public_weight": public_weight,
                "owned_weight": 1.0 - public_weight,
                "output_path": str(output_path),
                "correlation_with_incumbent": float(
                    np.corrcoef(prediction, incumbent)[0, 1]
                ),
                "delta_std": float((prediction - incumbent).std()),
                "rows": len(ids),
                "finite": bool(np.isfinite(prediction).all()),
            }
        )

    report = {
        "run_name": "current151_lower_public_grid",
        "incumbent_public_lb": 0.151,
        "incumbent_submission_ref": 56275973,
        "incumbent_weights": INCUMBENT_WEIGHTS,
        "reconstruction": reconstruction,
        "component_correlation": {
            "gpu_tpu": float(np.corrcoef(source["gpu"], source["tpu"])[0, 1]),
            "gpu_yangq": float(
                np.corrcoef(source["gpu"], source["yangq"])[0, 1]
            ),
            "tpu_yangq": float(
                np.corrcoef(source["tpu"], source["yangq"])[0, 1]
            ),
        },
        "outputs": outputs,
        "submission_status": "prepared_not_submitted",
    }
    METADATA_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




