"""Train the selected full-data quantile TabM and prepare B001 replacements."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as tabm
from audit_tabm_member_aggregation import predict_members
from exp_tabm_011_quantile_preprocessing import (
    QuantileBatchPreprocessor,
    fit_quantile_knots,
)
from exp_tabm_015_quantile_cosine_lr_curves import learning_rate_at
from export_full_tabm001_member_aggregations import load_test_only


OUTPUT_STEM = "tabm_quantile_coslr15_tree053r_blend75_fulltrain"
FULL_EPOCHS = 15
TABM_WEIGHT = 0.75
TREE_WEIGHT = 0.25


def unit(values: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0.0:
        raise AssertionError("Prediction norm must be finite and non-zero.")
    return values / norm, norm


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    tabm.SEED = 42
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "submissions" / OUTPUT_STEM
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    model_data, feature_columns, _, dropped_public_features = tabm.load_exp053r_data(
        project_dir
    )
    if set(model_data["month"].unique()) != set(range(71)):
        raise AssertionError("Full training data must contain months 0-70.")
    train_features = model_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    target = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    del model_data
    gc.collect()
    train_indices = np.arange(len(train_features), dtype=np.int64)

    preprocessing_path = run_dir / "quantile_preprocessing.npz"
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
        saved_columns = saved["feature_columns"].astype(str).tolist()
        if saved_columns != feature_columns:
            raise AssertionError("Saved quantile preprocessing schema differs.")
    else:
        knots, medians, missing_columns = fit_quantile_knots(
            train_features, train_indices, tabm.SEED
        )
        np.savez_compressed(
            preprocessing_path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray(feature_columns),
        )
    preprocessor = QuantileBatchPreprocessor(
        knots, medians, missing_columns, device
    )
    target_mean = float(target.mean(dtype=np.float64))
    target_std = float(target.std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_std).astype(np.float32)
    tabm.seed_everything(tabm.SEED)
    model = tabm.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tabm.LEARNING_RATE, weight_decay=tabm.WEIGHT_DECAY
    )
    checkpoint_path = run_dir / "training_checkpoint.pt"
    training_log_path = run_dir / "training_log.csv"
    start_epoch = 1
    logs: list[dict[str, float]] = []
    elapsed_before = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["next_epoch"])
        logs = list(checkpoint["logs"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        print(f"resuming full quantile TabM at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, FULL_EPOCHS + 1):
        learning_rate = learning_rate_at(epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        loss = tabm.train_one_epoch(
            model,
            optimizer,
            train_features,
            target_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        elapsed = elapsed_before + time.perf_counter() - started
        logs.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_mse": float(loss),
                "elapsed_seconds": elapsed,
            }
        )
        pd.DataFrame(logs).to_csv(training_log_path, index=False)
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "elapsed_seconds": elapsed,
            },
            checkpoint_path,
        )
        print(
            f"full quantile epoch={epoch:02d}/{FULL_EPOCHS:02d} "
            f"lr={learning_rate:.8f} loss={loss:.8f}",
            flush=True,
        )
    training_seconds = elapsed_before + time.perf_counter() - started
    model_path = project_dir / "outputs" / "models" / f"{OUTPUT_STEM}_tabm.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "input_dimension": preprocessor.output_dimension,
            "feature_columns": feature_columns,
            "target_mean": target_mean,
            "target_standard_deviation": target_std,
            "epochs": FULL_EPOCHS,
            "learning_rate_schedule": {
                "type": "cosine",
                "initial": 0.002,
                "minimum": 0.0001,
                "schedule_length_epochs": 20,
                "stopped_after_epoch": FULL_EPOCHS,
            },
        },
        model_path,
    )
    model = model.to(device)
    del train_features, target_scaled, train_indices, knots, medians
    gc.collect()
    torch.cuda.empty_cache()

    test_data, template, test_columns = load_test_only(project_dir)
    if test_columns != feature_columns:
        raise AssertionError("Train and test feature schemas differ.")
    test_features = test_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    del test_data
    gc.collect()
    members = predict_members(model, test_features, preprocessor, target_std)
    mean_prediction = members.mean(axis=1).astype(np.float64)
    trim1_prediction = np.sort(members, axis=1)[:, 1:-1].mean(axis=1).astype(
        np.float64
    )
    del test_features, members, model, preprocessor
    torch.cuda.empty_cache()
    gc.collect()

    existing = pd.read_feather(
        project_dir
        / "outputs"
        / "predictions"
        / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    if not np.array_equal(
        existing["sample_id"].to_numpy(), template["sample_id"].to_numpy()
    ):
        raise AssertionError("Existing XGBoost predictions are not aligned.")
    tree_prediction = existing["tree_prediction"].to_numpy(dtype=np.float64)
    tree_unit, tree_norm = unit(tree_prediction)
    source_predictions = {"mean": mean_prediction, "trim1": trim1_prediction}
    submission_dir = project_dir / "outputs" / "submissions"
    prediction_dir = project_dir / "outputs" / "predictions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    for path in (submission_dir, prediction_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for aggregation, tabm_prediction in source_predictions.items():
        tabm_unit, tabm_norm = unit(tabm_prediction)
        prediction = TABM_WEIGHT * tabm_unit + TREE_WEIGHT * tree_unit
        output_name = f"{OUTPUT_STEM}_{aggregation}"
        submission = template[["sample_id"]].copy()
        submission["prediction"] = prediction
        if submission.isna().any().any() or not np.isfinite(prediction).all():
            raise AssertionError(f"Invalid values in {output_name}.")
        submission_path = submission_dir / f"{output_name}.csv"
        prediction_path = prediction_dir / f"{output_name}_test.feather"
        submission.to_csv(submission_path, index=False)
        pd.DataFrame(
            {
                "sample_id": template["sample_id"].to_numpy(),
                "tabm_prediction": tabm_prediction,
                "tree_prediction": tree_prediction,
                "prediction": prediction,
            }
        ).to_feather(prediction_path)
        outputs[aggregation] = {
            "output_name": output_name,
            "submission_path": str(submission_path),
            "prediction_path": str(prediction_path),
            "tabm_norm": tabm_norm,
            "tree_norm": tree_norm,
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std(ddof=0)),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
        }

    metadata = {
        "output_stem": OUTPUT_STEM,
        "training_months": "0-70",
        "feature_count": len(feature_columns),
        "train_rows": int(len(target)),
        "test_rows": int(len(template)),
        "tabm_epochs": FULL_EPOCHS,
        "learning_rate_schedule": {
            "type": "smooth cosine decay",
            "initial": 0.002,
            "minimum": 0.0001,
            "schedule_length_epochs": 20,
            "stopped_after_epoch": FULL_EPOCHS,
        },
        "selection_basis": (
            "same epoch comparison across three forward windows; epoch 15 smooth "
            "improved 40-49 and 50-59 and was nearly neutral overall on 60-70"
        ),
        "blend": {"tabm_weight": TABM_WEIGHT, "xgboost_weight": TREE_WEIGHT},
        "training_seconds": training_seconds,
        "target_mean": target_mean,
        "target_standard_deviation": target_std,
        "model_path": str(model_path),
        "preprocessing_path": str(preprocessing_path),
        "dropped_public_features": dropped_public_features,
        "outputs": outputs,
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path = metadata_dir / f"{OUTPUT_STEM}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
