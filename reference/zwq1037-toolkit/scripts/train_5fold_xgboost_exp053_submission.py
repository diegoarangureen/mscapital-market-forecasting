"""Train a five-fold GPU bagging ensemble of EXP053 and create a submission."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    cosine_similarity_score,
    make_monthly_scores,
    save_model_safely,
)
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters
from train_full_xgboost_exp034_submission import add_selected_event_tables
from train_full_xgboost_exp051_submission import (
    add_gap_table,
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


EXPERIMENT_ID = "EXP-TREE-053-5FOLD"
OUTPUT_NAME = "xgboost_exp053_last15_5fold_gpu"
N_SPLITS = 5
FOLD_RANDOM_STATE = 42


def add_last15_table(
    project_dir: Path, split: str, data: pd.DataFrame
) -> pd.DataFrame:
    """Join the compact public-baseline final-15-second market block."""

    last15 = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / f"{split}_market_last15_baseline_features.feather"
    )
    output = data.merge(last15, on="sample_id", how="left", validate="one_to_one")
    del last15
    gc.collect()
    return output


def pairwise_prediction_cosines(predictions: list[np.ndarray]) -> list[dict]:
    """Summarize diversity between the five test-prediction vectors."""

    rows = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            rows.append(
                {
                    "fold_left": left,
                    "fold_right": right,
                    "cosine": cosine_similarity_score(
                        predictions[left], predictions[right]
                    ),
                }
            )
    return rows


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    public_features, dropped_public_features = selected_public_features(project_dir)

    train_data, base_features = load_features(project_dir, "train")
    test_data, test_base_features = load_features(project_dir, "test")
    if base_features != test_base_features:
        raise AssertionError("Train and test base feature schemas differ.")
    train_data = add_last15_table(
        project_dir,
        "train",
        add_gap_table(
            project_dir,
            "train",
            add_selected_event_tables(project_dir, "train", train_data),
        ),
    )
    test_data = add_last15_table(
        project_dir,
        "test",
        add_gap_table(
            project_dir,
            "test",
            add_selected_event_tables(project_dir, "test", test_data),
        ),
    )

    public_train = load_public_table(project_dir, "train", public_features).rename(
        columns={"target": "public_target"}
    )
    public_test = load_public_table(project_dir, "test", public_features)
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    train_data = (
        train_data.merge(public_train, on="sample_id", how="left", validate="one_to_one")
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    test_data = (
        test_data.merge(public_test, on="sample_id", how="left", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    target_difference = np.max(
        np.abs(
            train_data["public_target"].to_numpy(dtype=np.float64)
            - train_data["target"].to_numpy(dtype=np.float64)
        )
    )
    if target_difference > 1e-5:
        raise AssertionError("Public and official targets differ.")
    train_data = train_data.drop(columns="public_target")
    del public_train, public_test, labels
    gc.collect()

    # Use the complete raw-sequence RV already present in the public feature set.
    # The cached final-15-second numerator is complete for every sample.
    for data in [train_data, test_data]:
        data["x_rv_15_over_full"] = data["m_rv_15"] / (data["m_rv"] + 1e-8)

    feature_columns = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
        + public_features
        + LAST15_FEATURE_COLUMNS
    )
    if len(feature_columns) != 307 or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("EXP053 five-fold model must contain 307 unique features.")
    if {"sample_id", "month", "target"}.intersection(feature_columns):
        raise AssertionError("Identity, time, or target entered model features.")

    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if set(train_data["month"].unique()) != set(range(71)):
        raise AssertionError("Training data must cover labelled months 0 through 70.")
    if not np.array_equal(test_data["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test feature order differs from submission template.")

    target = train_data["target"].to_numpy(dtype=np.float32)
    global_target_mean = float(target.mean(dtype=np.float64))
    centered_target = target - global_target_mean
    fold_assignment = np.full(len(train_data), -1, dtype=np.int8)
    oof_prediction = np.full(len(train_data), np.nan, dtype=np.float64)
    test_predictions = []
    fold_metadata = []

    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    parameters["device"] = "cuda"
    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=FOLD_RANDOM_STATE,
    )
    model_dir = project_dir / "outputs" / "models" / OUTPUT_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    for fold_index, (fit_indices, holdout_indices) in enumerate(
        splitter.split(train_data), start=0
    ):
        fold_assignment[holdout_indices] = fold_index
        model = XGBRegressor(**parameters)
        fold_start = time.perf_counter()
        model.fit(
            train_data.iloc[fit_indices][feature_columns],
            centered_target[fit_indices],
        )
        training_seconds = time.perf_counter() - fold_start
        holdout_prediction = np.asarray(
            model.predict(train_data.iloc[holdout_indices][feature_columns]),
            dtype=np.float64,
        )
        test_prediction = np.asarray(
            model.predict(test_data[feature_columns]), dtype=np.float64
        )
        if not np.isfinite(holdout_prediction).all() or not np.isfinite(test_prediction).all():
            raise AssertionError(f"Fold {fold_index} produced non-finite predictions.")
        oof_prediction[holdout_indices] = holdout_prediction
        test_predictions.append(test_prediction)
        model_path = model_dir / f"fold_{fold_index}.json"
        save_model_safely(model, model_path)
        fold_metadata.append(
            {
                "fold": fold_index,
                "fit_rows": int(len(fit_indices)),
                "holdout_rows": int(len(holdout_indices)),
                "training_seconds": training_seconds,
                "holdout_cosine": cosine_similarity_score(
                    target[holdout_indices], holdout_prediction
                ),
                "model_path": str(model_path),
                "test_prediction_mean": float(test_prediction.mean()),
                "test_prediction_std": float(test_prediction.std(ddof=0)),
            }
        )
        print(
            f"fold={fold_index} fit_rows={len(fit_indices)} "
            f"seconds={training_seconds:.2f} "
            f"holdout_cosine={fold_metadata[-1]['holdout_cosine']:.8f}",
            flush=True,
        )
        del model, holdout_prediction, test_prediction
        gc.collect()

    if (fold_assignment < 0).any() or not np.isfinite(oof_prediction).all():
        raise AssertionError("Five-fold OOF coverage is incomplete.")
    prediction_matrix = np.vstack(test_predictions)
    prediction = prediction_matrix.mean(axis=0, dtype=np.float64)
    if prediction.shape != (len(template),) or not np.isfinite(prediction).all():
        raise AssertionError("Five-fold ensemble generated invalid test predictions.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate IDs or missing values.")

    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    prediction_dir = project_dir / "outputs" / "predictions"
    for output_dir in [submission_dir, metadata_dir, prediction_dir]:
        output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{OUTPUT_NAME}.csv"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    oof_path = prediction_dir / f"{OUTPUT_NAME}_oof.feather"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "sample_id": train_data["sample_id"].to_numpy(dtype=np.int32),
            "month": train_data["month"].to_numpy(dtype=np.int16),
            "target": target,
            "fold": fold_assignment,
            "prediction": oof_prediction,
        }
    ).to_feather(oof_path)

    exp053_config = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TREE-053"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    oof_frame = pd.DataFrame(
        {
            "month": train_data["month"].to_numpy(),
            "target": target,
            "prediction": oof_prediction,
        }
    )
    oof_monthly = make_monthly_scores(oof_frame)
    pairwise = pairwise_prediction_cosines(test_predictions)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "ensemble_type": "five-fold shuffled-row bagging; arithmetic mean",
        "fold_splitter": {
            "class": "sklearn.model_selection.KFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": FOLD_RANDOM_STATE,
            "note": "OOF is diagnostic only; temporal validation remains EXP053",
        },
        "training_months": "0-70; each model sees 80% of rows from all months",
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "public_selected_feature_count": len(public_features),
        "public_dropped_features": dropped_public_features,
        "training_target_mean": global_target_mean,
        "target_centered_without_adding_mean_back": True,
        "source_temporal_validation": {
            "official_60_70": exp053_config["overall_cosine"],
            "official_62_70_without_66": exp053_config["cosine_62_70_without_66"],
            "official_67_70": exp053_config["cosine_67_70"],
            "primary_monthly_std": exp053_config["primary_monthly_std"],
        },
        "random_oof_diagnostic": {
            "overall_cosine": cosine_similarity_score(target, oof_prediction),
            "monthly_population_std": float(oof_monthly["cosine"].std(ddof=0)),
            "monthly_min": float(oof_monthly["cosine"].min()),
            "monthly_max": float(oof_monthly["cosine"].max()),
        },
        "parameters": parameters,
        "xgboost_version": xgboost.__version__,
        "folds": fold_metadata,
        "total_training_and_prediction_seconds": time.perf_counter() - total_start,
        "test_prediction_pairwise_cosines": pairwise,
        "prediction_summary": {
            "mean": float(prediction.mean()),
            "std": float(prediction.std(ddof=0)),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
        },
        "submission_path": str(submission_path),
        "oof_prediction_path": str(oof_path),
        "model_directory": str(model_dir),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OOF diagnostic={metadata['random_oof_diagnostic']}", flush=True)
    print(f"prediction summary={metadata['prediction_summary']}", flush=True)
    print(f"submission={submission_path}", flush=True)


if __name__ == "__main__":
    main()
