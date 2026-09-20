"""Train the exact EXP-TREE-068 LightGBM on months 0-70 and predict test."""

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
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_full_tabm_tree_blend_submission import load_train_test


OUTPUT_NAME = "lightgbm068_fulltrain"


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    train, test, template, feature_columns, dropped_public_features = load_train_test(
        project_dir
    )
    if len(feature_columns) != 307 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Expected 307 unique EXP-TREE-068 features.")
    if not np.array_equal(test["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test IDs do not match submission template.")

    target = train["target"].to_numpy(dtype=np.float64, copy=True)
    target_mean = float(target.mean())
    centered_target = target - target_mean
    train_features = train[feature_columns].to_numpy(dtype=np.float32, copy=True)
    test_features = test[feature_columns].to_numpy(dtype=np.float32, copy=True)
    del train, test
    gc.collect()

    parameters = {
        "objective": "regression",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": 2,
        "verbosity": -1,
        "force_col_wise": True,
        "deterministic": True,
    }
    model = LGBMRegressor(**parameters)
    start = time.perf_counter()
    model.fit(train_features, centered_target)
    training_seconds = time.perf_counter() - start
    prediction = model.predict(test_features)
    if not np.isfinite(prediction).all():
        raise AssertionError("LightGBM test predictions contain non-finite values.")
    del train_features, test_features
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
        "source_experiment": "EXP-TREE-068",
        "training_months": "0-70",
        "train_rows": int(len(target)),
        "test_rows": int(len(template)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "public_dropped_features": dropped_public_features,
        "target_mean_removed": target_mean,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "training_device": "cpu-2-threads-exact-validation-backend",
        "lightgbm_version": lightgbm.__version__,
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
