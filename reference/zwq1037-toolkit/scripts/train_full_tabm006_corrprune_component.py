"""Train the full-data 293-feature correlation-pruned TabM component."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_full_tabm_tree_blend_submission as fulltrain


OUTPUT_NAME = "tabm006_corrprune_fulltrain"
FULL_EPOCHS = 15
EXPECTED_FEATURE_COUNT = 293


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
        project_dir / "data" / "interim" / "exp053r_correlated_feature_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_source"] != "training months 0-59 only":
        raise ValueError("Correlation pruning must come from training months only.")
    kept = set(selection["kept_features_in_gain_order"])
    selected_features = [name for name in feature_columns if name in kept]
    if len(selected_features) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_FEATURE_COUNT} selected features, got "
            f"{len(selected_features)}."
        )

    target = train_data["target"].to_numpy(dtype=np.float32, copy=True)
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
    prediction, details = fulltrain.train_tabm(
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
        "feature_count": len(selected_features),
        "feature_columns": selected_features,
        "public_dropped_features": dropped_public_features,
        "full_epochs": FULL_EPOCHS,
        "feature_selection": {
            "method": "training-only absolute Pearson correlation pruning",
            "threshold": selection["correlation_threshold"],
            "removed_feature_count": selection["removed_feature_count"],
            "selection_file": str(selection_path.relative_to(project_dir)),
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
