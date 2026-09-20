"""Export fulltrain TabM member mean and trimmed-mean test predictions."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tabm_001_exp053r_features import (
    BatchPreprocessor,
    EVAL_BATCH_SIZE,
    make_model,
)
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from train_5fold_xgboost_exp053_submission import add_last15_table
from train_full_xgboost_exp034_submission import add_selected_event_tables
from train_full_xgboost_exp051_submission import (
    add_gap_table,
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


OUTPUT_NAME = "tabm001_member_aggregations_fulltrain"


def load_test_only(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the exact 307-feature test table without loading training rows."""
    public_features, _ = selected_public_features(project_dir)
    test_data, base_features = load_features(project_dir, "test")
    test_data = add_last15_table(
        project_dir,
        "test",
        add_gap_table(
            project_dir,
            "test",
            add_selected_event_tables(project_dir, "test", test_data),
        ),
    )
    public_test = load_public_table(project_dir, "test", public_features)
    test_data = (
        test_data.merge(public_test, on="sample_id", how="left", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    test_data["x_rv_15_over_full"] = test_data["m_rv_15"] / (
        test_data["m_rv"] + 1.0e-8
    )
    feature_columns = (
        list(base_features)
        + list(TRANSACTION_FEATURE_COLUMNS)
        + list(ORDER_MULTI_FEATURE_COLUMNS)
        + list(GAP_FEATURE_COLUMNS)
        + list(public_features)
        + list(LAST15_FEATURE_COLUMNS)
    )
    if len(feature_columns) != 307 or len(set(feature_columns)) != 307:
        raise AssertionError("Expected 307 unique fulltrain features.")
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if not np.array_equal(
        test_data["sample_id"].to_numpy(), template["sample_id"].to_numpy()
    ):
        raise AssertionError("Test features do not match submission template order.")
    return test_data, template, feature_columns


@torch.inference_mode()
def predict_members(
    model,
    features: np.ndarray,
    preprocessor: BatchPreprocessor,
    target_standard_deviation: float,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(features), 16), dtype=np.float32)
    for start in range(0, len(features), EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, len(features))
        batch = preprocessor.transform(features[start:end])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(batch).squeeze(-1)
        output[start:end] = prediction.float().cpu().numpy()
    output *= target_standard_deviation
    return output


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
        project_dir
        / "outputs"
        / "models"
        / "tabm001_tree053r_blend75_fulltrain_tabm.pt",
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
        / "tabm001_tree053r_blend75_fulltrain_tabm_preprocessing.npz"
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
        / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    if not np.array_equal(existing["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Existing fulltrain predictions are not aligned.")
    reproduction_error = float(
        np.max(
            np.abs(
                mean_prediction
                - existing["tabm_prediction"].to_numpy(dtype=np.float32)
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
        "checkpoint": "outputs/models/tabm001_tree053r_blend75_fulltrain_tabm.pt",
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
