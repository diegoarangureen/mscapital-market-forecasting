"""Build the 0.150 aggressive blend with four upgraded owned models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factorized_transformer_lb_candidates import build_common_component


PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "aggressive_direct_owned_block_temporal4_equal38_common"
WEIGHTS = {
    "gpu_tabm142": 0.20,
    "tpu_tabm142": 0.12,
    "yangq_blend142": 0.32,
    "tabm385_temporal3x3": 0.126,
    "realmlp57": 0.054,
    "transformer_temporal3": 0.162,
    "gru_temporal3": 0.018,
}
PATHS = {
    "gpu_tabm142": PROJECT / "data/external/kaggle_public/bestwater_cos689/output/submission.csv",
    "tpu_tabm142": PROJECT / "data/external/kaggle_public/bestwater_tpu_v6/output/submission.csv",
    "yangq_blend142": PROJECT / "data/external/kaggle_public/yangq_lb142/output/submission.csv",
    "tabm385_temporal3x3": PROJECT / "outputs/submissions/tabm385_k32_temporal3fold_3seed.csv",
    "realmlp57": PROJECT / "outputs/submissions/realmlp379_rq16_temporal3fold_e8_fold_end57.csv",
    "transformer_temporal3": PROJECT / "outputs/submissions/standalone_factorized_transformer379_temporal3fold.csv",
    "gru_temporal3": PROJECT / "outputs/submissions/standalone_timeaware_three_stream_gru379_temporal3fold.csv",
}
OLD_PATHS = {
    "tabm385_temporal3x3": PROJECT / "outputs/submissions/tabm_relative319_xs40_order20_market6_k32_seed42_fulltrain.csv",
    "realmlp57": PROJECT / "outputs/submissions/realmlp379_rq16_corr095_e8_fulltrain.csv",
    "transformer_temporal3": PROJECT / "outputs/submissions/standalone_factorized_transformer379.csv",
    "gru_temporal3": PROJECT / "outputs/submissions/standalone_timeaware_three_stream_gru379.csv",
}


def zunit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def load(path: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns: {path}")
    if len(frame) != 647_896 or not frame["sample_id"].is_unique:
        raise AssertionError(f"Invalid rows or IDs: {path}")
    if ids is not None and not np.array_equal(frame["sample_id"].to_numpy(), ids):
        raise AssertionError(f"IDs do not align: {path}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Nonfinite predictions: {path}")
    return frame


def main() -> None:
    gru_metadata_path = (
        PROJECT
        / "outputs/submission_metadata/standalone_timeaware_three_stream_gru379_temporal3fold.json"
    )
    if not gru_metadata_path.exists():
        raise FileNotFoundError(f"Missing GRU selection metadata: {gru_metadata_path}")
    gru_metadata = json.loads(gru_metadata_path.read_text(encoding="utf-8"))
    PATHS["gru_temporal3"] = Path(gru_metadata["recommended_output_path"])
    missing = [str(path) for path in [*PATHS.values(), *OLD_PATHS.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing temporal blend inputs:\n" + "\n".join(missing))
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-12:
        raise AssertionError("Weights must sum to one")

    labels, template, groups, common = build_common_component()
    ids = template["sample_id"].to_numpy()
    frames = {name: load(path, ids) for name, path in PATHS.items()}
    old_frames = {name: load(path, ids) for name, path in OLD_PATHS.items()}
    source = {
        name: frame["prediction"].to_numpy(dtype=np.float64)
        for name, frame in frames.items()
    }
    before = sum(WEIGHTS[name] * zunit(source[name]) for name in WEIGHTS)
    prediction = before.copy()
    target_scale = float(labels["target"].to_numpy(dtype=np.float64).std())
    for group in np.unique(groups):
        mask = groups == group
        prediction[mask] = (
            prediction[mask] - prediction[mask].mean() + common[int(group)] / target_scale
        )
    if prediction.shape != (len(ids),) or not np.isfinite(prediction).all():
        raise AssertionError("Invalid final aggressive temporal prediction")

    incumbent_path = PROJECT / "outputs/submissions/aggressive_direct_owned_block_gru_equal38_common.csv"
    incumbent = load(incumbent_path, ids)
    conservative_path = PROJECT / "outputs/submissions/current150_temporal_model_upgrades_realmlp57.csv"
    conservative = load(conservative_path, ids) if conservative_path.exists() else None
    output_path = PROJECT / "outputs/submissions" / f"{RUN_NAME}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(output_path, index=False)

    replacement_correlations = {}
    for name, old_frame in old_frames.items():
        replacement_correlations[name] = float(
            np.corrcoef(
                old_frame["prediction"].to_numpy(dtype=np.float64), source[name]
            )[0, 1]
        )
    report = {
        "run_name": RUN_NAME,
        "base_public_lb": 0.150,
        "weights_before_common_component": WEIGHTS,
        "sources": {name: str(path) for name, path in PATHS.items()},
        "old_sources": {name: str(path) for name, path in OLD_PATHS.items()},
        "replacement_old_vs_new_test_correlation": replacement_correlations,
        "evidence": {
            "tabm_old_public_lb": 0.134,
            "tabm_temporal_public_lb": 0.140,
            "realmlp_old_public_lb": 0.139,
            "realmlp57_local_62_70_without_66": 0.15559188020185222,
            "realmlp_historical_local_62_70_without_66": 0.15466833910919217,
            "transformer_old_public_lb": 0.145,
            "gru_old_public_lb": 0.140,
        },
        "candidate": {
            "rows": len(ids),
            "finite": True,
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "correlation_with_aggressive_lb0150": float(
                np.corrcoef(prediction, incumbent["prediction"].to_numpy(dtype=np.float64))[0, 1]
            ),
            "correlation_with_conservative_temporal": (
                float(
                    np.corrcoef(
                        prediction,
                        conservative["prediction"].to_numpy(dtype=np.float64),
                    )[0, 1]
                )
                if conservative is not None
                else None
            ),
        },
        "output_path": str(output_path),
        "submission_status": "prepared_not_submitted",
        "submission_policy": "second and final daily submission if V12 evidence supports GRU inclusion",
    }
    metadata_path = PROJECT / "outputs/submission_metadata" / f"{RUN_NAME}.json"
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
