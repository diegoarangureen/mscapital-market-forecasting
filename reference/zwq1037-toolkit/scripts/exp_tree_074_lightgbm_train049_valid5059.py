"""Train the fixed EXP067 LightGBM configuration on 0-49 and predict 50-59."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from exp_tabm_001_exp053r_features import cosine_score, load_exp053r_data


EXPERIMENT_ID = "EXP-TREE-074-LIGHTGBM800-TRAIN049-VALID5059"


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "validation_predictions.feather"
    result_path = run_dir / "result.json"
    if prediction_path.exists() and result_path.exists():
        print(result_path.read_text(encoding="utf-8"), flush=True)
        return

    model_data, feature_columns, _, _ = load_exp053r_data(project_dir)
    train_mask = model_data["month"].to_numpy() <= 49
    valid_mask = (
        (model_data["month"].to_numpy() >= 50)
        & (model_data["month"].to_numpy() <= 59)
    )
    raw_target = model_data.loc[train_mask, "target"].to_numpy(dtype=np.float64)
    target_mean = float(raw_target.mean())
    parameters = {
        "objective": "regression",
        "n_estimators": 800,
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
    started = time.perf_counter()
    model.fit(
        model_data.loc[train_mask, feature_columns], raw_target - target_mean
    )
    training_seconds = time.perf_counter() - started
    prediction = np.asarray(
        model.predict(model_data.loc[valid_mask, feature_columns]), dtype=np.float64
    )
    validation = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    validation["prediction"] = prediction
    validation.to_feather(prediction_path)
    joblib.dump(model, run_dir / "model.joblib")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "train_months": "0-49",
        "validation_months": "50-59",
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "feature_count": len(feature_columns),
        "target_mean": target_mean,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "validation_cosine": cosine_score(
            validation["target"].to_numpy(dtype=np.float64), prediction
        ),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
