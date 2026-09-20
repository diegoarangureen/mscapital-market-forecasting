"""Run the public Salute-Rib high-feature LightGBM under our month-60 validation."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import assert_prediction_alignment
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


SOURCE_URL = "https://www.kaggle.com/code/sweetyheehee/lightgbm-baseline-salute-rib"


def load_public_namespace(project_dir: Path) -> dict:
    """Execute the saved public notebook code without invoking its submission main()."""

    notebook_path = (
        project_dir
        / "data"
        / "interim"
        / "public_notebooks"
        / "lgb_salute"
        / "lightgbm-baseline-salute-rib.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    os.environ["DATA_DIR"] = str(project_dir / "data" / "raw")
    os.environ["VALID_START_MONTH"] = "60"
    os.environ["N_THREADS"] = "16"
    namespace = {"__name__": "public_salute_lightgbm"}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    namespace["require_runtime"]()
    return namespace


def load_sorted_labels(project_dir: Path) -> pd.DataFrame:
    """Match the notebook's sample-id ordering while using our local label location."""

    return (
        pd.read_feather(
            project_dir / "data" / "raw" / "label.feather",
            columns=["sample_id", "month", "target"],
        )
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    public = load_public_namespace(project_dir)
    labels = load_sorted_labels(project_dir)
    sample_ids = labels["sample_id"].to_numpy(dtype=np.int64)
    months = labels["month"].to_numpy(dtype=np.int16)
    target = labels["target"].to_numpy(dtype=np.float32)
    train_mask = months < 60
    valid_mask = ~train_mask

    print("Building the public Salute-Rib feature set from raw event files...", flush=True)
    build_start = time.perf_counter()
    features = public["build_features"]("train", sample_ids)
    feature_names = list(features.keys())
    feature_seconds = time.perf_counter() - build_start
    print(
        f"public_features={len(feature_names)}, feature_seconds={feature_seconds:.1f}",
        flush=True,
    )
    x_train = public["make_matrix"](features, feature_names, train_mask)
    x_valid = public["make_matrix"](features, feature_names, valid_mask)
    del features
    gc.collect()

    parameters = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.025,
        "num_leaves": 48,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "max_bin": 63,
        "force_col_wise": True,
        "num_threads": 16,
        "verbose": -1,
        "seed": 7,
    }
    lightgbm = public["lgb"]
    train_dataset = lightgbm.Dataset(
        x_train,
        label=target[train_mask],
        params=parameters,
        free_raw_data=True,
    )
    valid_dataset = lightgbm.Dataset(
        x_valid,
        label=target[valid_mask],
        reference=train_dataset,
        params=parameters,
        free_raw_data=True,
    )
    train_dataset.construct()
    valid_dataset.construct()
    del x_train
    gc.collect()

    training_start = time.perf_counter()
    model = lightgbm.train(
        parameters,
        train_dataset,
        num_boost_round=3500,
        valid_sets=[valid_dataset],
        valid_names=["valid"],
        feval=public["lgb_cos_metric"],
        callbacks=[
            lightgbm.early_stopping(180, first_metric_only=False),
            lightgbm.log_evaluation(period=100),
        ],
    )
    training_seconds = time.perf_counter() - training_start
    predictions = model.predict(x_valid, num_iteration=model.best_iteration)
    del x_valid, train_dataset, valid_dataset
    gc.collect()

    validation_rows = labels.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].reset_index(drop=True)
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-037_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id="EXP-TREE-048",
        description=(
            "public Salute-Rib raw-event LightGBM reproduced with validation start changed "
            "from month 56 to month 60"
        ),
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_names,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-037",
        prediction_iteration=model.best_iteration,
        save_model=False,
    )
    model_path = project_dir / "outputs" / "models" / "exp-tree-048.txt"
    model_path.write_text(
        model.model_to_string(num_iteration=model.best_iteration), encoding="utf-8"
    )
    metadata.update(
        {
            "source_url": SOURCE_URL,
            "source_notebook": str(
                project_dir
                / "data"
                / "interim"
                / "public_notebooks"
                / "lgb_salute"
                / "lightgbm-baseline-salute-rib.ipynb"
            ),
            "public_code_changes": [
                "VALID_START_MONTH: 56 -> 60",
                "N_THREADS: 2 -> 16 (runtime only)",
                "skip test feature build and submission generation",
                "use local label.feather location",
            ],
            "feature_build_seconds": feature_seconds,
            "model_path": str(model_path),
        }
    )
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-048"
        / "config.json"
    )
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
