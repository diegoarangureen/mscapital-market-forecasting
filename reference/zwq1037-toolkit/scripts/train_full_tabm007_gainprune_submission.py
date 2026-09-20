"""Train full-data 277-feature gain-pruned TabM and create a Kaggle submission."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_full_tabm_tree_blend_submission as fulltrain


OUTPUT_NAME = "tabm007_gainprune_fulltrain"
FULL_EPOCHS = 15
EXPECTED_FEATURE_COUNT = 277


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
    selection_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TREE-064"
        / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_source"] != "EXP-TREE-053R training-only total_gain":
        raise ValueError("Gain pruning must come from training-only feature importance.")
    removed = set(selection["removed_features"])
    selected_features = [name for name in feature_columns if name not in removed]
    if len(selected_features) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_FEATURE_COUNT} selected features, got "
            f"{len(selected_features)}."
        )

    target = train_data["target"].to_numpy(dtype=np.float32, copy=True)
    # 分批装载逻辑由已有全量特征流程负责；这里只保留 277 列的连续数组。
    # Existing loaders control memory; retain only dense arrays for the 277 columns.
    train_features = train_data[selected_features].to_numpy(dtype=np.float32, copy=True)
    test_features = test_data[selected_features].to_numpy(dtype=np.float32, copy=True)
    del train_data, test_data
    gc.collect()
    print(
        f"Loaded train={len(target):,}, test={len(template):,}, "
        f"features={len(selected_features)}",
        flush=True,
    )

    fulltrain.OUTPUT_NAME = OUTPUT_NAME
    fulltrain.FULL_EPOCHS = FULL_EPOCHS
    prediction, tabm_details = fulltrain.train_tabm(
        project_dir,
        train_features,
        test_features,
        target,
        selected_features,
        run_dir,
        torch.device("cuda"),
    )
    del train_features, test_features
    gc.collect()
    if not np.isfinite(prediction).all():
        raise AssertionError("Submission prediction contains non-finite values.")

    submission = template[["sample_id"]].copy()
    submission["prediction"] = prediction
    if submission["sample_id"].duplicated().any() or submission.isna().any().any():
        raise AssertionError("Submission contains duplicate IDs or missing values.")
    submission_dir = project_dir / "outputs" / "submissions"
    prediction_dir = project_dir / "outputs" / "predictions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    for directory in (submission_dir, prediction_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{OUTPUT_NAME}.csv"
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "sample_id": template["sample_id"].to_numpy(),
            "prediction": prediction,
        }
    ).to_feather(prediction_path)

    validation_config = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-007-GAINPRUNE"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    metadata = {
        "output_name": OUTPUT_NAME,
        "training_months": "0-70",
        "train_rows": int(len(target)),
        "test_rows": int(len(template)),
        "feature_count": len(selected_features),
        "feature_columns": selected_features,
        "public_dropped_features": dropped_public_features,
        "full_epochs": FULL_EPOCHS,
        "feature_selection": {
            "selection_source": selection["selection_source"],
            "drop_fraction": selection["drop_fraction"],
            "removed_feature_count": selection["drop_count"],
            "removed_total_gain_share": selection["removed_total_gain_share"],
            "selection_file": str(selection_path.relative_to(project_dir)),
        },
        "local_validation": {
            key: validation_config[key]
            for key in [
                "overall_cosine",
                "cosine_62_70_without_66",
                "cosine_67_70",
                "cosine_60_64",
                "primary_monthly_std",
                "primary_monthly_worst",
                "primary_monthly_q25",
                "passes_core_generalization_gates",
                "passes_strict_stability_gates",
            ]
        },
        "tabm": tabm_details,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "submission_path": str(submission_path),
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
    print(f"submission={submission_path}", flush=True)
    print(json.dumps(metadata["prediction_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
