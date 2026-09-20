"""Export fulltrain cosine-TabM mean and trimmed member aggregations."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_tabm_001_exp053r_features import BatchPreprocessor, make_model
from export_full_tabm001_member_aggregations import load_test_only, predict_members


OUTPUT_NAME = "tabm003_cosine_member_aggregations_fulltrain"


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    project_dir = Path(__file__).resolve().parents[1]
    test_data, template, feature_columns = load_test_only(project_dir)
    checkpoint = torch.load(
        project_dir / "outputs" / "models" / "tabm003_cosine_fulltrain_tabm.pt",
        map_location="cpu",
        weights_only=False,
    )
    if feature_columns != checkpoint["feature_columns"]:
        raise AssertionError("Checkpoint and test feature schemas differ.")
    features = test_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    del test_data
    gc.collect()

    preprocessing = np.load(
        project_dir
        / "outputs"
        / "models"
        / "tabm003_cosine_fulltrain_tabm_preprocessing.npz"
    )
    device = torch.device("cuda")
    preprocessor = BatchPreprocessor(
        preprocessing["medians"],
        preprocessing["means"],
        preprocessing["standard_deviations"],
        preprocessing["missing_columns"],
        device,
    )
    model = make_model(int(checkpoint["input_dimension"]), device)
    model.load_state_dict(checkpoint["state_dict"])
    members = predict_members(
        model,
        features,
        preprocessor,
        float(checkpoint["target_standard_deviation"]),
    )
    del features, model
    torch.cuda.empty_cache()
    gc.collect()

    mean_prediction = members.mean(axis=1)
    sorted_members = np.sort(members, axis=1)
    trim1_prediction = sorted_members[:, 1:-1].mean(axis=1)
    trim2_prediction = sorted_members[:, 2:-2].mean(axis=1)
    existing = pd.read_feather(
        project_dir
        / "outputs"
        / "predictions"
        / "tabm003_cosine_fulltrain_test.feather"
    )
    if not np.array_equal(
        existing["sample_id"].to_numpy(), template["sample_id"].to_numpy()
    ):
        raise AssertionError("Existing fulltrain predictions are not aligned.")
    reproduction_error = float(
        np.max(
            np.abs(
                mean_prediction
                - existing["prediction"].to_numpy(dtype=np.float32)
            )
        )
    )

    prediction_dir = project_dir / "outputs" / "predictions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    prediction_path = prediction_dir / f"{OUTPUT_NAME}_test.feather"
    metadata_path = metadata_dir / f"{OUTPUT_NAME}.json"
    pd.DataFrame(
        {
            "sample_id": template["sample_id"].to_numpy(),
            "mean_prediction": mean_prediction,
            "trim1_prediction": trim1_prediction,
            "trim2_prediction": trim2_prediction,
        }
    ).to_feather(prediction_path)
    metadata = {
        "output_name": OUTPUT_NAME,
        "checkpoint": "outputs/models/tabm003_cosine_fulltrain_tabm.pt",
        "test_rows": int(len(template)),
        "member_count": int(members.shape[1]),
        "mean_reproduction_max_absolute_error": reproduction_error,
        "aggregation_uses_labels": False,
        "prediction_path": str(prediction_path),
        "statistics": {
            name: {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "l2_norm": float(np.linalg.norm(values)),
            }
            for name, values in {
                "mean": mean_prediction,
                "trim1": trim1_prediction,
                "trim2": trim2_prediction,
            }.items()
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
