"""Build LB candidates that introduce the full-data Factorized Transformer.

The script prepares several predeclared variants. It never submits them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
REF = PROJECT / "data/interim/kaggle_kernels/relative319_xs_tabm_notebook"
sys.path.insert(0, str(REF))
import run_extracted as recipe
from train_full_gru_embedding_candidate import load_test_relative319


SOURCE_PATHS = {
    "current138": PROJECT / "outputs/submissions/current138_xs40_tabm20.csv",
    "gpu_tabm142": PROJECT / "data/external/kaggle_public/bestwater_cos689/output/submission.csv",
    "tpu_tabm142": PROJECT / "data/external/kaggle_public/bestwater_tpu_v6/output/submission.csv",
    "yangq_blend142": PROJECT / "data/external/kaggle_public/yangq_lb142/output/submission.csv",
    "public_realmlp131": PROJECT / "data/external/kaggle_public/yunsu_realmlp/output/submission.csv",
    "old_transformer137": PROJECT / "outputs/submissions/joint_transformer_hybrid_seed42e5_seed137e4.csv",
    "own_tabm379": PROJECT / "outputs/submissions/tabm_relative319_xs40_order20_seed42_fulltrain.csv",
    "own_realmlp379": PROJECT / "outputs/submissions/realmlp379_rq16_yunsu_e8_fulltrain.csv",
    "own_realmlp379_corr095": PROJECT / "outputs/submissions/realmlp379_rq16_corr095_e8_fulltrain.csv",
}


VARIANTS = {
    "factorized_half_replace": {
        "current138": 0.15,
        "gpu_tabm142": 0.20,
        "tpu_tabm142": 0.12,
        "yangq_blend142": 0.32,
        "public_realmlp131": 0.05,
        "old_transformer137": 0.08,
        "factorized_transformer": 0.08,
    },
    "factorized_full_replace": {
        "current138": 0.15,
        "gpu_tabm142": 0.20,
        "tpu_tabm142": 0.12,
        "yangq_blend142": 0.32,
        "public_realmlp131": 0.05,
        "factorized_transformer": 0.16,
    },
    "factorized_half_replace_new_realmlp": {
        "current138": 0.15,
        "gpu_tabm142": 0.20,
        "tpu_tabm142": 0.12,
        "yangq_blend142": 0.32,
        "own_realmlp379": 0.05,
        "old_transformer137": 0.08,
        "factorized_transformer": 0.08,
    },
    "factorized_full_replace_new_realmlp": {
        "current138": 0.15,
        "gpu_tabm142": 0.20,
        "tpu_tabm142": 0.12,
        "yangq_blend142": 0.32,
        "own_realmlp379": 0.05,
        "factorized_transformer": 0.16,
    },
    "factorized_full_replace_corr095_realmlp": {
        "current138": 0.15,
        "gpu_tabm142": 0.20,
        "tpu_tabm142": 0.12,
        "yangq_blend142": 0.32,
        "own_realmlp379_corr095": 0.05,
        "factorized_transformer": 0.16,
    },
    "controllable_stack36": {
        "gpu_tabm142": 0.20,
        "tpu_tabm142": 0.12,
        "yangq_blend142": 0.32,
        "own_tabm379": 0.162,
        "own_realmlp379": 0.0612,
        "factorized_transformer": 0.1368,
    },
}


def zunit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()


def summarize(values: np.ndarray, groups: np.ndarray, target: np.ndarray | None = None):
    rows, targets, keys = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        group_values = np.asarray(values[mask], dtype=np.float64)
        rows.append(np.r_[np.nanmean(group_values, axis=0), np.nanstd(group_values, axis=0)])
        keys.append(int(group))
        if target is not None:
            targets.append(float(target[mask].mean()))
    return np.asarray(rows), np.asarray(targets), np.asarray(keys)


def build_common_component():
    labels = pd.read_feather(
        PROJECT / "data/raw/label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    train = np.load(PROJECT / "data/interim/kaggle_relative319_dev/features.npy", mmap_mode="r")
    columns = json.loads(
        (PROJECT / "data/interim/kaggle_relative319_dev/feature_columns.json").read_text(encoding="utf-8")
    )
    feature_indices = [columns.index(column) for column in recipe.TOP_FEATURES]
    train_x, train_y, _ = summarize(
        train[:, feature_indices], labels["month"].to_numpy(), labels["target"].to_numpy(float)
    )
    medians = np.nanmedian(train_x, axis=0)
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=100.0).fit(scaler.transform(train_x), train_y)

    test, template, test_columns = load_test_relative319(PROJECT)
    assert test_columns == columns
    groups = (np.arange(len(test), dtype=np.int64) * 38 // len(test)).astype(np.int16)
    test_x, _, group_keys = summarize(test[:, feature_indices], groups)
    test_x = np.where(np.isfinite(test_x), test_x, medians)
    common = dict(zip(group_keys, model.predict(scaler.transform(test_x))))
    return labels, template, groups, common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorized", type=Path, required=True)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()

    source_paths = dict(SOURCE_PATHS)
    source_paths["factorized_transformer"] = args.factorized.resolve()
    selected_variants = VARIANTS if args.variant == "all" else {args.variant: VARIANTS[args.variant]}
    needed_sources = sorted({name for weights in selected_variants.values() for name in weights})
    frames = {name: pd.read_csv(source_paths[name]) for name in needed_sources}

    labels, template, groups, common = build_common_component()
    sample_ids = template["sample_id"].to_numpy()
    for name, frame in frames.items():
        assert list(frame.columns) == ["sample_id", "prediction"], name
        assert np.array_equal(frame["sample_id"].to_numpy(), sample_ids), name
        assert np.isfinite(frame["prediction"].to_numpy()).all(), name

    source_predictions = {name: frames[name]["prediction"].to_numpy(float) for name in needed_sources}
    target_scale = float(labels["target"].to_numpy(float).std())
    reports = {}
    for variant, weights in selected_variants.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-12, variant
        before_common = sum(weights[name] * zunit(source_predictions[name]) for name in weights)
        prediction = before_common.copy()
        for group in np.unique(groups):
            mask = groups == group
            prediction[mask] = prediction[mask] - prediction[mask].mean() + common[int(group)] / target_scale

        output_path = PROJECT / f"outputs/submissions/{variant}_equal38_common.csv"
        no_common_path = PROJECT / f"outputs/submissions/{variant}_no_common.csv"
        pd.DataFrame({"sample_id": sample_ids, "prediction": prediction}).to_csv(output_path, index=False)
        pd.DataFrame({"sample_id": sample_ids, "prediction": before_common}).to_csv(no_common_path, index=False)

        names = list(weights)
        correlation = np.corrcoef([source_predictions[name] for name in names])
        report = {
            "run_name": variant,
            "weights_before_common_component": weights,
            "sources": {name: str(source_paths[name]) for name in names},
            "test_prediction_correlation": correlation.tolist(),
            "before_vs_after_common": float(np.corrcoef(before_common, prediction)[0, 1]),
            "prediction": {
                "mean": float(prediction.mean()),
                "std": float(prediction.std()),
                "min": float(prediction.min()),
                "max": float(prediction.max()),
                "finite": bool(np.isfinite(prediction).all()),
            },
            "output_path": str(output_path),
            "no_common_output_path": str(no_common_path),
            "submission_status": "prepared_not_submitted",
        }
        metadata_path = PROJECT / f"outputs/submission_metadata/{variant}_equal38_common.json"
        metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[variant] = report

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

