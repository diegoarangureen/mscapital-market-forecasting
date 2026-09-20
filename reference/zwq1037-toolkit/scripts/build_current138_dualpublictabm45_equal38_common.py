"""Build the leading candidate: current best + two public TabMs + 38-block common component."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
REF = PROJECT / "data" / "interim" / "kaggle_kernels" / "relative319_xs_tabm_notebook"
sys.path.insert(0, str(REF))
import run_extracted as recipe
from train_full_gru_embedding_candidate import load_test_relative319

RUN = "current138_dualpublictabm45_equal38_common"
CURRENT_WEIGHT = 0.55
GPU_WEIGHT = 0.225
TPU_WEIGHT = 0.225
ALPHA = 100.0
GROUPS = 38

def zunit(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std()

def summarize(values, groups, target=None):
    rows, targets, keys = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        group_values = np.asarray(values[mask], dtype=np.float64)
        rows.append(np.r_[np.nanmean(group_values, axis=0), np.nanstd(group_values, axis=0)])
        keys.append(int(group))
        if target is not None:
            targets.append(float(target[mask].mean()))
    return np.asarray(rows), np.asarray(targets), np.asarray(keys)

def main():
    labels = pd.read_feather(PROJECT / "data" / "raw" / "label.feather", columns=["sample_id", "month", "target"]).sort_values("sample_id").reset_index(drop=True)
    train = np.load(PROJECT / "data" / "interim" / "kaggle_relative319_dev" / "features.npy", mmap_mode="r")
    columns = json.loads((PROJECT / "data" / "interim" / "kaggle_relative319_dev" / "feature_columns.json").read_text(encoding="utf-8"))
    feature_indices = [columns.index(column) for column in recipe.TOP_FEATURES]

    train_x, train_y, _ = summarize(train[:, feature_indices], labels["month"].to_numpy(), labels["target"].to_numpy(float))
    medians = np.nanmedian(train_x, axis=0)
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=ALPHA).fit(scaler.transform(train_x), train_y)

    test, template, test_columns = load_test_relative319(PROJECT)
    assert test_columns == columns
    groups = (np.arange(len(test), dtype=np.int64) * GROUPS // len(test)).astype(np.int16)
    test_x, _, group_keys = summarize(test[:, feature_indices], groups)
    test_x = np.where(np.isfinite(test_x), test_x, medians)
    common = dict(zip(group_keys, model.predict(scaler.transform(test_x))))

    source_paths = {
        "current138": PROJECT / "outputs" / "submissions" / "current135_hybrid2_conv1_transformer30.csv",
        "gpu_tabm": PROJECT / "data" / "external" / "kaggle_public" / "bestwater_cos689" / "output" / "submission.csv",
        "tpu_tabm": PROJECT / "data" / "external" / "kaggle_public" / "bestwater_tpu_v6" / "output" / "submission.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in source_paths.items()}
    for name, frame in frames.items():
        assert list(frame.columns) == ["sample_id", "prediction"], name
        assert np.array_equal(frame["sample_id"].to_numpy(), template["sample_id"].to_numpy()), name
        assert np.isfinite(frame["prediction"].to_numpy()).all(), name

    source_predictions = {name: frame["prediction"].to_numpy(float) for name, frame in frames.items()}
    prediction = (
        CURRENT_WEIGHT * zunit(source_predictions["current138"])
        + GPU_WEIGHT * zunit(source_predictions["gpu_tabm"])
        + TPU_WEIGHT * zunit(source_predictions["tpu_tabm"])
    )
    before_common = prediction.copy()
    target_scale = float(labels["target"].to_numpy(float).std())
    for group in np.unique(groups):
        mask = groups == group
        prediction[mask] = prediction[mask] - prediction[mask].mean() + common[int(group)] / target_scale

    output_path = PROJECT / "outputs" / "submissions" / f"{RUN}.csv"
    pd.DataFrame({"sample_id": frames["current138"]["sample_id"], "prediction": prediction}).to_csv(output_path, index=False)
    report = {
        "run_name": RUN,
        "sources": {name: str(path) for name, path in source_paths.items()},
        "source_public_scores": {"current138": 0.138, "gpu_tabm": 0.142, "tpu_tabm": 0.142},
        "weights_before_common_component": {"current138": CURRENT_WEIGHT, "gpu_tabm": GPU_WEIGHT, "tpu_tabm": TPU_WEIGHT},
        "method": "z-score each source, blend 55/22.5/22.5, then replace each of 38 equal sequential group means with Ridge(alpha=100) input-only common target component / full target std",
        "rows": int(len(prediction)),
        "group_count": GROUPS,
        "group_size_min": int(np.bincount(groups).min()),
        "group_size_max": int(np.bincount(groups).max()),
        "source_prediction_correlations": {
            "current_gpu": float(np.corrcoef(source_predictions["current138"], source_predictions["gpu_tabm"])[0, 1]),
            "current_tpu": float(np.corrcoef(source_predictions["current138"], source_predictions["tpu_tabm"])[0, 1]),
            "gpu_tpu": float(np.corrcoef(source_predictions["gpu_tabm"], source_predictions["tpu_tabm"])[0, 1]),
        },
        "candidate_correlations": {
            "candidate_vs_current138": float(np.corrcoef(prediction, source_predictions["current138"])[0, 1]),
            "candidate_vs_gpu_tabm": float(np.corrcoef(prediction, source_predictions["gpu_tabm"])[0, 1]),
            "candidate_vs_tpu_tabm": float(np.corrcoef(prediction, source_predictions["tpu_tabm"])[0, 1]),
            "before_vs_after_common": float(np.corrcoef(before_common, prediction)[0, 1]),
        },
        "prediction": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "finite": bool(np.isfinite(prediction).all()),
        },
        "common_normalized_min": float(min(common.values()) / target_scale),
        "common_normalized_max": float(max(common.values()) / target_scale),
        "local_validation": {
            "50_59": {"score": 0.164436},
            "62_70_excluding_66": {"score": 0.163490},
            "delta_vs_current138": {"50_59": 0.008440, "62_70_excluding_66": 0.006228},
        },
        "submission_status": "prepared_not_submitted",
        "output_path": str(output_path),
    }
    metadata_path = PROJECT / "outputs" / "submission_metadata" / f"{RUN}.json"
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

