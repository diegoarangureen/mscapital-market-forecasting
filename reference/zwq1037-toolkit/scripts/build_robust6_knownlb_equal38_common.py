"""Build robust six-source candidate guided by known LB scores, correlations, and forum production weights."""
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

RUN = "robust6_knownlb_equal38_common"
WEIGHTS = {
    "current138": 0.15,
    "gpu_tabm142": 0.20,
    "tpu_tabm142": 0.12,
    "yangq_blend142": 0.32,
    "realmlp131": 0.05,
    "transformer137": 0.16,
}
KNOWN_SCORES = {
    "current138": 0.138,
    "gpu_tabm142": 0.142,
    "tpu_tabm142": 0.142,
    "yangq_blend142": 0.142,
    "realmlp131": 0.131,
    "transformer137": 0.137,
}
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
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-12:
        raise AssertionError("weights must sum to one")
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
        "current138": PROJECT / "outputs" / "submissions" / "current138_xs40_tabm20.csv",
        "gpu_tabm142": PROJECT / "data" / "external" / "kaggle_public" / "bestwater_cos689" / "output" / "submission.csv",
        "tpu_tabm142": PROJECT / "data" / "external" / "kaggle_public" / "bestwater_tpu_v6" / "output" / "submission.csv",
        "yangq_blend142": PROJECT / "data" / "external" / "kaggle_public" / "yangq_lb142" / "output" / "submission.csv",
        "realmlp131": PROJECT / "data" / "external" / "kaggle_public" / "yunsu_realmlp" / "output" / "submission.csv",
        "transformer137": PROJECT / "outputs" / "submissions" / "joint_transformer_hybrid_seed42e5_seed137e4.csv",
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
    correlation = np.corrcoef([source_predictions[name] for name in names])
    score_vector = np.asarray([KNOWN_SCORES[name] for name in names])
    weight_vector = np.asarray([WEIGHTS[name] for name in names])
    implied_score = float(weight_vector @ score_vector / np.sqrt(weight_vector @ correlation @ weight_vector))
    report = {
        "run_name": RUN,
        "sources": {name: str(path) for name, path in source_paths.items()},
        "source_public_scores": KNOWN_SCORES,
        "weights_before_common_component": WEIGHTS,
        "known_score_correlation_implied_score_before_common": implied_score,
        "test_prediction_correlation": correlation.tolist(),
        "method": "z-score each source, robust six-source blend, then replace each of 38 equal sequential group means with Ridge(alpha=100) input-only common target component / full target std",
        "evidence": {
            "forum_production": "Reported 0.147 with external YQ+RM; 0.148 after adding 15% CNN-Transformer.",
            "our_two_oof_source_validation": "Current + two public TabMs + common improved both preferred windows; independent Transformer has known LB0.137.",
            "new_public152_xs75_tabm": "Rejected as weak: 0.13353 / 0.13126 on preferred windows.",
            "multistream_gru": "Rejected as weak: 0.15460 on first window vs current 0.15632.",
        },
        "rows": int(len(prediction)),
        "group_count": GROUPS,
        "group_size_min": int(np.bincount(groups).min()),
        "group_size_max": int(np.bincount(groups).max()),
        "candidate_correlations": {name: float(np.corrcoef(prediction, source_predictions[name])[0, 1]) for name in names},
        "before_vs_after_common": float(np.corrcoef(no_common_prediction, prediction)[0, 1]),
        "prediction": {"mean": float(prediction.mean()), "std": float(prediction.std()), "min": float(prediction.min()), "max": float(prediction.max()), "finite": bool(np.isfinite(prediction).all())},
        "common_normalized_min": float(min(common.values()) / target_scale),
        "common_normalized_max": float(max(common.values()) / target_scale),
        "submission_status": "prepared_not_submitted",
        "output_path": str(output_path),
        "no_common_output_path": str(no_common_path),
    }
    (PROJECT / "outputs" / "submission_metadata" / f"{RUN}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

