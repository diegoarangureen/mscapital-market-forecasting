"""Fit the validated 379-feature XGBoost on all labelled months."""
from __future__ import annotations
import json, os, time
from pathlib import Path
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely

PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "xgboost_relative319_xs40_order20_fulltrain"

def main():
    train = np.load(PROJECT / "data/interim/our379_reference_cache/features.npy", mmap_mode="r")
    target = np.load(PROJECT / "data/interim/our379_reference_cache/targets.npy", mmap_mode="r").astype(np.float64)
    train_ids = np.load(PROJECT / "data/interim/kaggle_relative319_dev/sample_ids.npy", mmap_mode="r")
    test = np.load(PROJECT / "data/interim/our379_reference_test_cache/features.npy", mmap_mode="r")
    test_ids = np.load(PROJECT / "data/interim/our379_reference_test_cache/sample_ids.npy", mmap_mode="r")
    template = pd.read_csv(PROJECT / "data/raw/submission.csv")
    if train.shape != (len(target), 379) or test.shape != (len(test_ids), 379):
        raise AssertionError("Expected 379-feature caches")
    if len(train_ids) != len(target) or not np.array_equal(test_ids, template["sample_id"].to_numpy()):
        raise AssertionError("Sample IDs do not align")
    params = dict(objective="reg:squarederror", n_estimators=800, learning_rate=0.03, max_depth=5, min_child_weight=100.0, reg_lambda=1.0, reg_alpha=0.0, subsample=1.0, colsample_bytree=0.8, tree_method="hist", max_bin=255, random_state=42, n_jobs=2, verbosity=0, device="cuda")
    model = XGBRegressor(**params)
    centered_target = target - target.mean()
    started = time.perf_counter()
    model.fit(train, centered_target)
    seconds = time.perf_counter() - started
    prediction = np.asarray(model.predict(test), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError("Invalid predictions")
    submission_path = PROJECT / "outputs/submissions" / f"{RUN_NAME}.csv"
    prediction_path = PROJECT / "outputs/predictions" / f"{RUN_NAME}_test.feather"
    model_path = PROJECT / "outputs/models" / f"{RUN_NAME}.json"
    pd.DataFrame({"sample_id": test_ids, "prediction": prediction}).to_csv(submission_path, index=False)
    pd.DataFrame({"sample_id": test_ids, "prediction": prediction}).to_feather(prediction_path)
    save_model_safely(model, model_path)
    metadata = {"run_name":RUN_NAME,"training_months":"0-70","train_rows":len(train),"test_rows":len(test),"feature_count":379,"validation_62_70_no66":0.14102067530316598,"parameters":params,"xgboost_version":xgboost.__version__,"training_seconds":seconds,"prediction_std":float(prediction.std()),"submission_path":str(submission_path),"prediction_path":str(prediction_path),"model_path":str(model_path),"submission_status":"prepared_not_uploaded"}
    meta_path = PROJECT / "outputs/submission_metadata" / f"{RUN_NAME}.json"
    meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
if __name__ == "__main__":
    main()