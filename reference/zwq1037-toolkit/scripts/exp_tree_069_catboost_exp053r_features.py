"""Evaluate GPU CatBoost on the complete 307-feature EXP053R schema."""

from __future__ import annotations

import json
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from exp_tabm_001_exp053r_features import load_exp053r_data
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
    save_model_safely,
)
from exp_tree_027_031_xgboost_capacity_regularization import evaluate_and_save


EXPERIMENT_ID = "EXP-TREE-069"
CANDIDATE_ITERATIONS = [400, 800]


def cosine_score(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction))
    )


def model_parameters(iterations: int) -> dict:
    return {
        "loss_function": "RMSE",
        "iterations": iterations,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 5.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_strength": 0.5,
        "border_count": 128,
        "random_seed": 42,
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.80,
        "thread_count": 2,
        "allow_writing_files": False,
        "verbose": 100,
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data, feature_columns, _, dropped_public_features = load_exp053r_data(
        project_dir
    )
    target = model_data["target"].to_numpy(dtype=np.float64)
    months = model_data["month"].to_numpy()

    internal_train = months <= 49
    internal_valid = (months >= 50) & (months <= 59)
    internal_target_mean = float(target[internal_train].mean())
    internal_centered_target = target[internal_train] - internal_target_mean
    maximum_iterations = max(CANDIDATE_ITERATIONS)
    internal_parameters = model_parameters(maximum_iterations)
    internal_model = CatBoostRegressor(**internal_parameters)
    print("Training internal GPU CatBoost to 800 trees", flush=True)
    internal_started = time.perf_counter()
    internal_model.fit(
        model_data.loc[internal_train, feature_columns], internal_centered_target
    )
    internal_seconds = time.perf_counter() - internal_started
    internal_rows = []
    for iterations in CANDIDATE_ITERATIONS:
        prediction = np.asarray(
            internal_model.predict(
                model_data.loc[internal_valid, feature_columns],
                ntree_end=iterations,
            ),
            dtype=np.float64,
        )
        score = cosine_score(target[internal_valid], prediction)
        internal_rows.append({"iterations": iterations, "cosine_50_59": score})
        print(f"iterations={iterations} cosine_50_59={score:.10f}", flush=True)
    pd.DataFrame(internal_rows).to_csv(
        run_dir / "iteration_selection.csv", index=False
    )
    selected_iterations = int(
        max(internal_rows, key=lambda row: row["cosine_50_59"])["iterations"]
    )
    del internal_model

    formal_train = months <= 59
    formal_valid = months >= 60
    formal_target_mean = float(target[formal_train].mean())
    formal_centered_target = target[formal_train] - formal_target_mean
    parameters = model_parameters(selected_iterations)
    model = CatBoostRegressor(**parameters)
    print(
        f"Training formal GPU CatBoost with {selected_iterations} trees", flush=True
    )
    formal_started = time.perf_counter()
    model.fit(model_data.loc[formal_train, feature_columns], formal_centered_target)
    prediction = np.asarray(
        model.predict(model_data.loc[formal_valid, feature_columns]),
        dtype=np.float64,
    )
    formal_seconds = time.perf_counter() - formal_started
    validation_rows = model_data.loc[
        formal_valid, ["sample_id", "month", "target"]
    ].copy()
    baseline = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
    )
    assert_prediction_alignment(baseline, validation_rows)
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "GPU CatBoost depth 7 on the complete 307-feature EXP053R schema; "
            "400/800 iterations selected on months 50-59"
        ),
        model=None,
        predictions=prediction,
        validation_rows=validation_rows,
        baseline_predictions=baseline,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=formal_seconds,
        baseline_id="EXP-TREE-053R",
        save_model=False,
    )
    model_path = project_dir / "outputs" / "models" / f"{EXPERIMENT_ID.lower()}.cbm"
    model.save_model(model_path)
    metadata.pop("xgboost_version", None)
    metadata.update(
        {
            "model_family": "CatBoost",
            "catboost_version": catboost.__version__,
            "training_device": "cuda",
            "selected_iterations": selected_iterations,
            "iteration_selection": internal_rows,
            "internal_training_seconds": internal_seconds,
            "formal_training_seconds": formal_seconds,
            "formal_target_mean": formal_target_mean,
            "model_path": str(model_path),
            "public_dropped_features": dropped_public_features,
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
