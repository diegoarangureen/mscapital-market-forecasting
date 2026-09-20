"""Train the full-data 307-feature cosine-loss TabM component."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp_tabm_001_exp053r_features as tabm_base
import exp_tabm_003_cosine_loss as cosine_experiment
import train_full_tabm_tree_blend_submission as fulltrain


OUTPUT_NAME = "tabm003_cosine_fulltrain"
FULL_EPOCHS = 11
EXPECTED_FEATURE_COUNT = 307


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
    train_data, test_data, template, feature_columns, dropped_public_features = (
        fulltrain.load_train_test(project_dir)
    )
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_columns)}."
        )
    target = train_data["target"].to_numpy(dtype=np.float32, copy=True)
    train_features = train_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    test_features = test_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    del train_data, test_data
    gc.collect()
    print(
        f"Loaded train={len(target):,}, test={len(template):,}, "
        f"features={len(feature_columns)}",
        flush=True,
    )

    # 复用已验证的逐成员 batch cosine loss 与 30 轮余弦学习率轨迹。
    # Reuse the validated per-member batch cosine loss and 30-epoch LR trajectory.
    fulltrain.OUTPUT_NAME = OUTPUT_NAME
    fulltrain.FULL_EPOCHS = FULL_EPOCHS
    fulltrain.LEARNING_RATE = cosine_experiment.INITIAL_LEARNING_RATE
    fulltrain.WEIGHT_DECAY = cosine_experiment.WEIGHT_DECAY
    fulltrain.train_one_epoch = cosine_experiment.train_one_epoch_cosine
    tabm_base.SEED = 42
    prediction, details = fulltrain.train_tabm(
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

    prediction_dir = project_dir / "outputs" / "predictions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    pd.DataFrame(
        {"sample_id": template["sample_id"].to_numpy(), "prediction": prediction}
    ).to_feather(prediction_path)
    metadata = {
        "output_name": OUTPUT_NAME,
        "training_months": "0-70",
        "train_rows": int(len(target)),
        "test_rows": int(len(template)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "public_dropped_features": dropped_public_features,
        "selected_full_epochs": FULL_EPOCHS,
        "selection_source": "EXP-TABM-003-COSINE validation selected epoch 11",
        "loss": "mean per-member batch cosine loss",
        "learning_rate_schedule": {
            "type": "cosine decay",
            "initial": cosine_experiment.INITIAL_LEARNING_RATE,
            "minimum": cosine_experiment.MINIMUM_LEARNING_RATE,
            "schedule_length_epochs": cosine_experiment.MAX_EPOCHS,
        },
        "tabm": details,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "prediction_path": str(prediction_path),
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
    print(json.dumps(metadata["prediction_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
