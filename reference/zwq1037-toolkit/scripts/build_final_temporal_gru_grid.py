"""Build final candidates using the validated temporal-three-fold GRU."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factorized_transformer_lb_candidates import build_common_component


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "outputs" / "submissions"
METADATA_PATH = PROJECT / "outputs/submission_metadata/final_temporal_gru_grid.json"
EXPECTED_ROWS = 647_896
PATHS = {
    "gpu_tabm": PROJECT / "data/external/kaggle_public/bestwater_cos689/output/submission.csv",
    "tpu_tabm": PROJECT / "data/external/kaggle_public/bestwater_tpu_v6/output/submission.csv",
    "yangq": PROJECT / "data/external/kaggle_public/yangq_lb142/output/submission.csv",
    "tabm_temporal": PROJECT / "outputs/submissions/tabm385_k32_temporal3fold_3seed.csv",
    "realmlp57": PROJECT / "outputs/submissions/realmlp379_rq16_temporal3fold_e8_fold_end57.csv",
    "transformer_old": PROJECT / "outputs/submissions/standalone_factorized_transformer379.csv",
    "gru_temporal": PROJECT / "outputs/submissions/standalone_timeaware_three_stream_gru379_temporal3fold.csv",
    "gru_old": PROJECT / "outputs/submissions/standalone_timeaware_three_stream_gru379.csv",
    "base_no_gru": PROJECT / "outputs/submissions/current150_tabmfrac1p5_realmlpfrac0p5.csv",
}


def zunit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = values.std()
    if standard_deviation == 0.0:
        raise ValueError("Cannot normalize a constant prediction vector.")
    return (values - values.mean()) / standard_deviation


def load(path: Path, expected_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if len(frame) != EXPECTED_ROWS or not frame["sample_id"].is_unique:
        raise AssertionError(f"Invalid submission rows: {path}")
    ids = frame["sample_id"].to_numpy()
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise AssertionError(f"Sample IDs differ: {path}")
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError(f"Nonfinite prediction: {path}")
    return ids, prediction


def add_common_component(
    prediction: np.ndarray,
    groups: np.ndarray,
    common: dict,
    target_scale: float,
) -> np.ndarray:
    output = prediction.copy()
    for group in np.unique(groups):
        mask = groups == group
        output[mask] = output[mask] - output[mask].mean() + common[int(group)] / target_scale
    return output


def centered_delta(delta: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = delta.copy()
    for group in np.unique(groups):
        mask = groups == group
        output[mask] -= output[mask].mean()
    return output


def main() -> None:
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing blend inputs:\n" + "\n".join(missing))
    labels, template, groups, common = build_common_component()
    ids = template["sample_id"].to_numpy()
    source = {}
    for name, path in PATHS.items():
        _, prediction = load(path, ids)
        source[name] = prediction
    normalized = {name: zunit(values) for name, values in source.items()}
    target_scale = float(labels["target"].to_numpy(dtype=np.float64).std())
    outputs = []

    # Keep the public block at 64%; search only a small, interpretable owned block.
    for tabm_weight, realmlp_weight, gru_weight in itertools.product(
        (0.12, 0.15, 0.18), (0.03, 0.05), (0.03, 0.05, 0.07)
    ):
        transformer_weight = 0.36 - tabm_weight - realmlp_weight - gru_weight
        if transformer_weight < 0.06:
            continue
        weights = {
            "gpu_tabm": 0.20,
            "tpu_tabm": 0.12,
            "yangq": 0.32,
            "tabm_temporal": tabm_weight,
            "realmlp57": realmlp_weight,
            "transformer_old": transformer_weight,
            "gru_temporal": gru_weight,
        }
        if abs(sum(weights.values()) - 1.0) > 1e-12:
            raise AssertionError("Direct weights do not sum to one.")
        before_common = sum(weights[name] * normalized[name] for name in weights)
        prediction = add_common_component(before_common, groups, common, target_scale)
        run_name = (
            f"final_gru_direct_t{int(tabm_weight * 100):02d}_"
            f"r{int(realmlp_weight * 100):02d}_"
            f"x{int(transformer_weight * 100):02d}_"
            f"g{int(gru_weight * 100):02d}"
        )
        output_path = OUTPUT_DIR / f"{run_name}.csv"
        pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(output_path, index=False)
        outputs.append(
            {
                "run_name": run_name,
                "kind": "direct_owned_block",
                "weights": weights,
                "output_path": str(output_path),
                "correlation_with_base_no_gru": float(
                    np.corrcoef(prediction, source["base_no_gru"])[0, 1]
                ),
            }
        )

    # Add only the temporal-GRU improvement direction to the strongest no-GRU base.
    gru_upgrade_direction = normalized["gru_temporal"] - normalized["gru_old"]
    for upgrade_weight in (0.018, 0.03, 0.05, 0.07):
        delta = centered_delta(upgrade_weight * gru_upgrade_direction, groups)
        prediction = source["base_no_gru"] + delta
        weight_label = str(upgrade_weight).replace("0.", "")
        run_name = f"final_gru_residual_w{weight_label}"
        output_path = OUTPUT_DIR / f"{run_name}.csv"
        pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(output_path, index=False)
        outputs.append(
            {
                "run_name": run_name,
                "kind": "temporal_upgrade_residual",
                "upgrade_weight": upgrade_weight,
                "output_path": str(output_path),
                "correlation_with_base_no_gru": float(
                    np.corrcoef(prediction, source["base_no_gru"])[0, 1]
                ),
            }
        )

    component_names = ["tabm_temporal", "realmlp57", "transformer_old", "gru_temporal", "gru_old"]
    report = {
        "run_name": "final_temporal_gru_grid",
        "gru_evidence": {
            "strict_validation_equal3": 0.16066187874296348,
            "old_vs_temporal_test_correlation": 0.959977705407524,
        },
        "component_names": component_names,
        "component_correlation": np.corrcoef([source[name] for name in component_names]).tolist(),
        "outputs": outputs,
        "submission_status": "diagnostic_candidates_not_submitted",
    }
    METADATA_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
