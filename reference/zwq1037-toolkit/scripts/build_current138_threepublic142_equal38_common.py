"""Build four-source candidate from three known LB0.142 public models plus current best."""
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

RUN = "current138_threepublic142_equal38_common"
WEIGHTS = {"current138": 0.35, "gpu_tabm": 0.175, "tpu_tabm": 0.175, "yangq_blend": 0.30}
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
        "yangq_blend": PROJECT / "data" / "external" / "kaggle_public" / "yangq_lb142" / "output" / "submission.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in source_paths.items()}
    for name, frame in frames.items():
        assert list(frame.columns) == ["sample_id", "prediction"], name
        assert np.array_equal(frame["sample_id"].to_numpy(), template["sample_id"].to_numpy()), name
        assert np.isfinite(frame["prediction"].to_numpy()).all(), name

    source_predictions = {name: frame["prediction"].to_numpy(float) for name, frame in frames.items()}
    prediction = sum(WEIGHTS[name] * zunit(source_predictions[name]) for name in WEIGHTS)
    no_common_prediction = prediction.copy()
    target_scale = float(labels["target"].to_numpy(float).std())
    for group in np.unique(groups):
        mask = groups == group
        prediction[mask] = prediction[mask] - prediction[mask].mean() + common[int(group)] / target_scale

    output_path = PROJECT / "outputs" / "submissions" / f"{RUN}.csv"
    no_common_path = PROJECT / "outputs" / "submissions" / f"{RUN}_no_common.csv"
    pd.DataFrame({"sample_id": frames["current138"]["sample_id"], "prediction": prediction}).to_csv(output_path, index=False)
    pd.DataFrame({"sample_id": frames["current138"]["sample_id"], "prediction": no_common_prediction}).to_csv(no_common_path, index=False)

    names = list(WEIGHTS)
    matrix = np.corrcoef([source_predictions[name] for name in names])
    known_scores = np.asarray([0.138, 0.142, 0.142, 0.142])
    weight_vector = np.asarray([WEIGHTS[name] for name in names])
    implied_score = float(weight_vector @ known_scores / np.sqrt(weight_vector @ matrix @ weight_vector))
    report = {
        "run_name": RUN,
        "sources": {name: str(path) for name, path in source_paths.items()},
        "source_public_scores": {"current138": 0.138, "gpu_tabm": 0.142, "tpu_tabm": 0.142, "yangq_blend": 0.142},
        "weights_before_common_component": WEIGHTS,
        "test_prediction_correlation": matrix.tolist(),
        "known_score_correlation_implied_score_before_common": implied_score,
        "method": "z-score each source, blend 35/17.5/17.5/30, then replace each of 38 equal sequential group means with Ridge(alpha=100) input-only common target component / full target std",
        "rows": int(len(prediction)),
        "group_count": GROUPS,
        "group_size_min": int(np.bincount(groups).min()),
        "group_size_max": int(np.bincount(groups).max()),
        "candidate_correlations": {name: float(np.corrcoef(prediction, source_predictions[name])[0, 1]) for name in names},
        "before_vs_after_common": float(np.corrcoef(no_common_prediction, prediction)[0, 1]),
        "prediction": {"mean": float(prediction.mean()), "std": float(prediction.std()), "min": float(prediction.min()), "max": float(prediction.max()), "finite": bool(np.isfinite(prediction).all())},
        "common_normalized_min": float(min(common.values()) / target_scale),
        "common_normalized_max": float(max(common.values()) / target_scale),
        "validation_evidence": {
            "three_oof_source_candidate_55_22.5_22.5_with_common": {"50_59": 0.164436, "62_70_excluding_66": 0.163490},
            "yangq_blend": "No OOF supplied; full submission is explicitly reported LB0.142. Weight selected using its known LB and test prediction correlation matrix.",
        },
        "submission_status": "prepared_not_submitted",
        "output_path": str(output_path),
        "no_common_output_path": str(no_common_path),
    }
    (PROJECT / "outputs" / "submission_metadata" / f"{RUN}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

