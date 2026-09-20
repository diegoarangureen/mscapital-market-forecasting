"""Train full-data EXP-TABM-001 and EXP-TREE-053R, then save their 75/25 blend."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost
from xgboost import XGBRegressor

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tabm_001_exp053r_features import (
    BATCH_SIZE,
    D_BLOCK,
    DROPOUT,
    K,
    LEARNING_RATE,
    N_BLOCKS,
    SEED,
    WEIGHT_DECAY,
    BatchPreprocessor,
    atomic_torch_save,
    fit_preprocessor,
    make_model,
    predict,
    seed_everything,
    train_one_epoch,
)
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters
from train_5fold_xgboost_exp053_submission import add_last15_table
from train_full_xgboost_exp034_submission import add_selected_event_tables
from train_full_xgboost_exp051_submission import (
    add_gap_table,
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


OUTPUT_NAME = "tabm001_tree053r_blend75_fulltrain"
TABM_WEIGHT = 0.75
TREE_WEIGHT = 0.25
FULL_EPOCHS = 15


def load_train_test(
    project_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """装载与本地验证完全相同的307个特征。 / Load the exact validated 307 features."""
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
    if target_difference > 1.0e-5:
        raise AssertionError("Public and official targets differ.")
    train_data = train_data.drop(columns="public_target")

    for data in (train_data, test_data):
        data["x_rv_15_over_full"] = data["m_rv_15"] / (data["m_rv"] + 1.0e-8)

    feature_columns = (
        list(base_features)
        + list(TRANSACTION_FEATURE_COLUMNS)
        + list(ORDER_MULTI_FEATURE_COLUMNS)
        + list(GAP_FEATURE_COLUMNS)
        + list(public_features)
        + list(LAST15_FEATURE_COLUMNS)
    )
    if len(feature_columns) != 307 or len(set(feature_columns)) != 307:
        raise AssertionError("Full-data blend must contain 307 unique features.")
    if set(train_data["month"].unique()) != set(range(71)):
        raise AssertionError("Training data must cover months 0 through 70.")

    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if not np.array_equal(
        test_data["sample_id"].to_numpy(), template["sample_id"].to_numpy()
    ):
        raise AssertionError("Test feature order differs from the submission template.")
    del public_train, public_test, labels
    gc.collect()
    return train_data, test_data, template, feature_columns, dropped_public_features


def train_tree(
    project_dir: Path,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    centered_target: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """训练与 EXP053R 匹配的单个全量 XGBoost。 / Train the matching full XGBoost."""
    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    parameters["device"] = "cuda"
    model = XGBRegressor(**parameters)
    started_at = time.perf_counter()
    model.fit(train_data[feature_columns], centered_target)
    prediction = np.asarray(model.predict(test_data[feature_columns]), dtype=np.float64)
    elapsed_seconds = time.perf_counter() - started_at
    if not np.isfinite(prediction).all():
        raise AssertionError("Tree produced non-finite test predictions.")

    model_path = project_dir / "outputs" / "models" / f"{OUTPUT_NAME}_tree.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_safely(model, model_path)
    details = {
        "parameters": parameters,
        "elapsed_seconds": elapsed_seconds,
        "model_path": str(model_path),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(ddof=0)),
    }
    del model
    gc.collect()
    return prediction, details


def train_tabm(
    project_dir: Path,
    train_features: np.ndarray,
    test_features: np.ndarray,
    target: np.ndarray,
    feature_columns: list[str],
    run_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """用固定15轮训练全量 TabM。 / Train full TabM for the selected 15 epochs."""
    train_indices = np.arange(len(train_features), dtype=np.int64)
    test_indices = np.arange(len(test_features), dtype=np.int64)
    medians, means, standard_deviations, missing_columns = fit_preprocessor(
        train_features, train_indices
    )
    preprocessor = BatchPreprocessor(
        medians, means, standard_deviations, missing_columns, device
    )
    target_mean = float(target.mean(dtype=np.float64))
    target_standard_deviation = float(target.std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_standard_deviation).astype(np.float32)

    preprocessing_path = project_dir / "outputs" / "models" / f"{OUTPUT_NAME}_tabm_preprocessing.npz"
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preprocessing_path,
        medians=medians,
        means=means,
        standard_deviations=standard_deviations,
        missing_columns=missing_columns,
        feature_columns=np.asarray(feature_columns),
    )

    seed_everything(SEED)
    model = make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    checkpoint_path = run_dir / "fulltrain_tabm_checkpoint.pt"
    log_path = run_dir / "fulltrain_tabm_epochs.csv"
    start_epoch = 1
    logs: list[dict] = []
    elapsed_before_resume = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["next_epoch"])
        logs = list(checkpoint["logs"])
        elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        print(f"Resuming full TabM at epoch {start_epoch}", flush=True)

    started_at = time.perf_counter()
    for epoch in range(start_epoch, FULL_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        elapsed_seconds = elapsed_before_resume + time.perf_counter() - started_at
        logs.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        pd.DataFrame(logs).to_csv(log_path, index=False)
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "elapsed_seconds": elapsed_seconds,
            },
            checkpoint_path,
        )
        print(
            f"Full TabM epoch {epoch:02d}/{FULL_EPOCHS:02d}: loss={train_loss:.6f}",
            flush=True,
        )

    elapsed_seconds = elapsed_before_resume + time.perf_counter() - started_at
    prediction = predict(model, test_features, test_indices, preprocessor).astype(np.float64)
    prediction *= target_standard_deviation
    if not np.isfinite(prediction).all():
        raise AssertionError("TabM produced non-finite test predictions.")

    model_path = project_dir / "outputs" / "models" / f"{OUTPUT_NAME}_tabm.pt"
    model_cpu = model.cpu()
    torch.save(
        {
            "state_dict": model_cpu.state_dict(),
            "model_parameters": {
                "k": K,
                "n_blocks": N_BLOCKS,
                "d_block": D_BLOCK,
                "dropout": DROPOUT,
            },
            "input_dimension": preprocessor.output_dimension,
            "feature_columns": feature_columns,
            "target_mean": target_mean,
            "target_standard_deviation": target_standard_deviation,
            "full_epochs": FULL_EPOCHS,
        },
        model_path,
    )
    details = {
        "parameters": {
            "k": K,
            "n_blocks": N_BLOCKS,
            "d_block": D_BLOCK,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": FULL_EPOCHS,
            "precision": "bfloat16",
        },
        "elapsed_seconds": elapsed_seconds,
        "model_path": str(model_path),
        "preprocessing_path": str(preprocessing_path),
        "input_dimension": preprocessor.output_dimension,
        "missing_indicator_count": int(len(missing_columns)),
        "target_mean": target_mean,
        "target_standard_deviation": target_standard_deviation,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(ddof=0)),
    }
    return prediction, details


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "submissions" / OUTPUT_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data, test_data, template, feature_columns, dropped_public_features = load_train_test(
        project_dir
    )
    target = train_data["target"].to_numpy(dtype=np.float32, copy=True)
    target_mean = float(target.mean(dtype=np.float64))
    centered_target = target - target_mean
    print(
        f"Loaded train={len(train_data):,}, test={len(test_data):,}, "
        f"features={len(feature_columns)}",
        flush=True,
    )

    print("Training full EXP053R tree branch", flush=True)
    tree_prediction, tree_details = train_tree(
        project_dir,
        train_data,
        test_data,
        feature_columns,
        centered_target,
    )
    print(
        f"Tree completed in {tree_details['elapsed_seconds']:.1f}s", flush=True
    )

    # 转为连续 float32 数组后释放 DataFrame，降低全量 TabM 训练的内存压力。
    # Convert to dense arrays, then release DataFrames to reduce peak RAM.
    train_features = train_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    test_features = test_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    del train_data, test_data, centered_target
    gc.collect()

    print("Training full TabM branch", flush=True)
    tabm_prediction, tabm_details = train_tabm(
        project_dir,
        train_features,
        test_features,
        target,
        feature_columns,
        run_dir,
        torch.device("cuda"),
    )
    del train_features, test_features
    gc.collect()

    tabm_norm = float(np.linalg.norm(tabm_prediction))
    tree_norm = float(np.linalg.norm(tree_prediction))
    blended_prediction = (
        TABM_WEIGHT * tabm_prediction / tabm_norm
        + TREE_WEIGHT * tree_prediction / tree_norm
    )
    if not np.isfinite(blended_prediction).all():
        raise AssertionError("Blend produced non-finite predictions.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = blended_prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate IDs or missing values.")

    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    prediction_dir = project_dir / "outputs" / "predictions"
    for directory in (submission_dir, metadata_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{OUTPUT_NAME}.csv"
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "sample_id": template["sample_id"].to_numpy(),
            "tabm_prediction": tabm_prediction,
            "tree_prediction": tree_prediction,
            "prediction": blended_prediction,
        }
    ).to_feather(prediction_path)

    validation_config = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-BLEND-001"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    metadata = {
        "output_name": OUTPUT_NAME,
        "training_months": "0-70",
        "train_rows": int(len(target)),
        "test_rows": int(len(template)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "public_dropped_features": dropped_public_features,
        "blend": {
            "tabm_weight": TABM_WEIGHT,
            "tree_weight": TREE_WEIGHT,
            "normalization": "each test prediction vector divided by its global L2 norm",
            "tabm_test_norm": tabm_norm,
            "tree_test_norm": tree_norm,
        },
        "local_validation": {
            "overall_cosine": validation_config["overall_cosine"],
            "cosine_62_70_without_66": validation_config["cosine_62_70_without_66"],
            "cosine_67_70": validation_config["cosine_67_70"],
            "cosine_60_64": validation_config["cosine_60_64"],
            "primary_monthly_std": validation_config["primary_monthly_std"],
            "primary_monthly_worst": validation_config["primary_monthly_worst"],
            "passes_core_generalization_gates": validation_config[
                "passes_core_generalization_gates"
            ],
            "passes_strict_stability_gates": validation_config[
                "passes_strict_stability_gates"
            ],
        },
        "tabm": tabm_details,
        "tree": tree_details,
        "xgboost_version": xgboost.__version__,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "submission_path": str(submission_path),
        "prediction_path": str(prediction_path),
        "prediction_summary": {
            "mean": float(blended_prediction.mean()),
            "std": float(blended_prediction.std(ddof=0)),
            "min": float(blended_prediction.min()),
            "max": float(blended_prediction.max()),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"submission={submission_path}", flush=True)
    print(json.dumps(metadata["prediction_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
