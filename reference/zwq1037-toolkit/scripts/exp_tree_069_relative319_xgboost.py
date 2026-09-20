"""Test the promoted relative add12 features in the EXP053R XGBoost component."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import (
    FOLDS,
    load_xgboost_prediction,
    unit,
)
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_027_031_xgboost_capacity_regularization import fit_model, model_parameters


EXPERIMENT_ID = "EXP-TREE-069-RELATIVE319"


def run_fold(
    project_dir: Path,
    run_dir: Path,
    fold_name: str,
    train_end: int,
    valid_start: int,
    valid_end: int,
    model_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "result.json"
    if result_path.exists():
        print(f"reusing completed {fold_name}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    train_mask = model_data["month"].to_numpy() <= train_end
    valid_mask = (
        (model_data["month"].to_numpy() >= valid_start)
        & (model_data["month"].to_numpy() <= valid_end)
    )
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_ids = model_data.loc[valid_mask, "sample_id"].to_numpy()
    validation_months = model_data.loc[valid_mask, "month"].to_numpy()
    validation_target = model_data.loc[valid_mask, "target"].to_numpy(dtype=np.float64)

    parameters = model_parameters(
        n_estimators=800,
        colsample_bytree=0.8,
        device="cuda",
        n_jobs=2,
    )
    model, training_seconds = fit_model(
        model_data,
        feature_columns,
        train_mask,
        centered_target,
        parameters,
    )
    candidate_xgb = np.asarray(
        model.predict(model_data.loc[valid_mask, feature_columns]), dtype=np.float64
    )
    baseline_xgb = load_xgboost_prediction(project_dir, fold_name, validation_ids)
    tabm_frame = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-016-RELATIVE-SCALE-ADD12"
        / fold_name
        / "validation_predictions.feather"
    )
    if not np.array_equal(tabm_frame["sample_id"].to_numpy(), validation_ids):
        raise AssertionError(f"TabM relative predictions are misaligned for {fold_name}.")
    relative_tabm = tabm_frame["tabm_mean"].to_numpy(dtype=np.float64)

    predictions = {
        "xgb_baseline307": baseline_xgb,
        "xgb_relative319": candidate_xgb,
        "blend_old_xgb": 0.75 * unit(relative_tabm) + 0.25 * unit(baseline_xgb),
        "blend_relative_xgb": 0.75 * unit(relative_tabm) + 0.25 * unit(candidate_xgb),
    }
    metrics = {
        name: evaluate(
            validation_target,
            prediction,
            validation_months,
            valid_start,
            valid_end,
        )
        for name, prediction in predictions.items()
    }
    for candidate_name, baseline_name in (
        ("xgb_relative319", "xgb_baseline307"),
        ("blend_relative_xgb", "blend_old_xgb"),
    ):
        for key in (
            "overall",
            "monthly_std",
            "monthly_worst",
            "monthly_q25",
            "without_month_66",
            "months_67_70",
        ):
            if key in metrics[candidate_name]:
                metrics[candidate_name][f"{key}_delta"] = (
                    metrics[candidate_name][key] - metrics[baseline_name][key]
                )

    prediction_frame = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "month": validation_months,
            "target": validation_target,
            "relative_tabm": relative_tabm,
            **predictions,
        }
    )
    prediction_frame.to_feather(fold_dir / "validation_predictions.feather")
    monthly = pd.DataFrame(
        {
            "month": list(range(valid_start, valid_end + 1)),
            **{
                name: [row["cosine"] for row in values["monthly"]]
                for name, values in metrics.items()
            },
        }
    )
    monthly.to_csv(fold_dir / "monthly_cosine.csv", index=False)
    summary = []
    for name, values in metrics.items():
        summary.append(
            {
                "candidate": name,
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"monthly", "first_half_months", "second_half_months"}
                },
            }
        )
    pd.DataFrame(summary).to_csv(fold_dir / "summary.csv", index=False)
    save_model_safely(model, fold_dir / "model.json")
    result = {
        "fold": fold_name,
        "train_months": f"0-{train_end}",
        "validation_months": f"{valid_start}-{valid_end}",
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "feature_count": len(feature_columns),
        "parameters": parameters,
        "training_seconds": training_seconds,
        "metrics": metrics,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    del model
    gc.collect()
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    model_data, baseline_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, model_data)
    feature_columns = [*baseline_columns, *RELATIVE_COLUMNS]
    if len(feature_columns) != 319 or len(set(feature_columns)) != 319:
        raise AssertionError("Expected exactly 319 unique features.")

    results = {}
    for fold_name, (train_end, valid_start, valid_end) in FOLDS.items():
        results[fold_name] = run_fold(
            project_dir,
            run_dir,
            fold_name,
            train_end,
            valid_start,
            valid_end,
            model_data,
            feature_columns,
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "single_variable_change": "add the promoted relative add12 group to EXP053R XGBoost",
        "baseline_feature_count": len(baseline_columns),
        "candidate_feature_count": len(feature_columns),
        "added_features": RELATIVE_COLUMNS,
        "folds": results,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
