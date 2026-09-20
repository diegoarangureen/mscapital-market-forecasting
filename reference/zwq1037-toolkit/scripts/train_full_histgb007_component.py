"""Train the 97-feature EXP-TREE-007 HistGB on months 0-70 and predict test."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


OUTPUT_NAME = "histgb007_fulltrain"


def load_split(project_dir: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    processed = project_dir / "data" / "processed"
    v1 = pd.read_feather(processed / f"{split}_market_features.feather")
    last60 = pd.read_feather(
        processed / f"{split}_market_last60_features_complete.feather"
    )
    level2 = pd.read_feather(processed / f"{split}_market_level2_features.feather")
    data = (
        v1.merge(last60, on="sample_id", how="left", validate="one_to_one")
        .merge(level2, on="sample_id", how="left", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    data["last60_row_count"] = data["last60_row_count"].fillna(0)
    base_features = v1.columns[1:].tolist() + last60.columns[1:].tolist()
    level2_features = level2.columns[1:].tolist()
    if set(base_features).intersection(level2_features):
        raise AssertionError("Level-two feature names overlap base features.")
    features = base_features + level2_features
    if len(features) != 97 or len(features) != len(set(features)):
        raise AssertionError("Expected 97 unique EXP-TREE-007 features.")
    del v1, last60, level2
    gc.collect()
    return data, features


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    train, train_features = load_split(project_dir, "train")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "target"],
    )
    train = train.merge(labels, on="sample_id", how="inner", validate="one_to_one")
    train = train.sort_values("sample_id").reset_index(drop=True)
    test, test_features = load_split(project_dir, "test")
    if train_features != test_features:
        raise AssertionError("Train and test HistGB feature schemas differ.")
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if not np.array_equal(test["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test IDs do not match submission template.")

    target = train["target"].to_numpy(dtype=np.float64, copy=True)
    target_mean = float(target.mean())
    centered_target = target - target_mean
    train_features_array = train[train_features].to_numpy(dtype=np.float32, copy=True)
    test_features_array = test[test_features].to_numpy(dtype=np.float32, copy=True)
    del train, test, labels
    gc.collect()

    parameters = {
        "loss": "squared_error",
        "learning_rate": 0.03,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
        "verbose": 0,
    }
    model = HistGradientBoostingRegressor(**parameters)
    start = time.perf_counter()
    model.fit(train_features_array, centered_target)
    training_seconds = time.perf_counter() - start
    prediction = model.predict(test_features_array)
    if not np.isfinite(prediction).all():
        raise AssertionError("HistGB test predictions contain non-finite values.")
    del train_features_array, test_features_array
    gc.collect()

    prediction_path = (
        project_dir / "outputs" / "predictions" / f"{OUTPUT_NAME}_test.feather"
    )
    model_path = project_dir / "outputs" / "models" / f"{OUTPUT_NAME}.joblib"
    metadata_path = (
        project_dir / "outputs" / "submission_metadata" / f"{OUTPUT_NAME}.json"
    )
    pd.DataFrame(
        {"sample_id": template["sample_id"].to_numpy(), "prediction": prediction}
    ).to_feather(prediction_path)
    joblib.dump(model, model_path)
    metadata = {
        "output_name": OUTPUT_NAME,
        "source_experiment": "EXP-TREE-007",
        "training_months": "0-70",
        "train_rows": int(len(target)),
        "test_rows": int(len(template)),
        "feature_count": len(train_features),
        "feature_columns": train_features,
        "target_mean_removed": target_mean,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        },
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "prediction_summary": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std(ddof=0)),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "l2_norm": float(np.linalg.norm(prediction)),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
