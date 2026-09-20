"""Corrected fulltrain EXP-TREE-007 using train feature order for test."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from train_full_histgb007_component import OUTPUT_NAME, load_split


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
    if set(train_features) != set(test_features):
        raise AssertionError("Train and test HistGB feature sets differ.")
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if not np.array_equal(test["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test IDs do not match submission template.")

    target = train["target"].to_numpy(dtype=np.float64, copy=True)
    target_mean = float(target.mean())
    centered_target = target - target_mean
    train_array = train[train_features].to_numpy(dtype=np.float32, copy=True)
    # 训练与测试源表列顺序不同，显式使用训练列顺序。
    # Source schemas differ in order, so explicitly apply the training order.
    test_array = test[train_features].to_numpy(dtype=np.float32, copy=True)
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
    model.fit(train_array, centered_target)
    training_seconds = time.perf_counter() - start
    prediction = model.predict(test_array)
    if not np.isfinite(prediction).all():
        raise AssertionError("HistGB test predictions contain non-finite values.")
    del train_array, test_array
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
        "test_columns_reordered_to_training_schema": True,
        "target_mean_removed": target_mean,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
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
