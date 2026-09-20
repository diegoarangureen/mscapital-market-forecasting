"""Export 16 TabM members and audit robust label-free aggregations."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_tabm_001_exp053r_features import (
    BatchPreprocessor,
    EVAL_BATCH_SIZE,
    load_exp053r_data,
    make_model,
)


EXPERIMENT_ID = "EXP-TABM-MEMBER-001"


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def evaluate(target: np.ndarray, prediction: np.ndarray, months: np.ndarray) -> dict:
    month_values = np.sort(np.unique(months))
    monthly = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    primary = monthly[(month_values >= 62) & (month_values != 66)]
    return {
        "overall": cosine(target, prediction),
        "no66": cosine(
            target[(months >= 62) & (months != 66)],
            prediction[(months >= 62) & (months != 66)],
        ),
        "recent": cosine(target[months >= 67], prediction[months >= 67]),
        "early": cosine(target[months <= 64], prediction[months <= 64]),
        "primary_std": float(primary.std(ddof=0)),
        "primary_worst": float(primary.min()),
        "primary_q25": float(np.quantile(primary, 0.25)),
    }


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
            batch_prediction = model(batch).squeeze(-1)
        output[start:end] = batch_prediction.float().cpu().numpy()
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
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        project_dir / "outputs" / "models" / "exp-tabm-001.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_data, feature_columns, _, _ = load_exp053r_data(project_dir)
    if feature_columns != checkpoint["feature_columns"]:
        raise AssertionError("Checkpoint and current feature schemas differ.")
    validation_mask = model_data["month"].to_numpy() >= 60
    validation_rows = model_data.loc[
        validation_mask, ["sample_id", "month", "target"]
    ].reset_index(drop=True)
    validation_features = model_data.loc[
        validation_mask, feature_columns
    ].to_numpy(dtype=np.float32, copy=True)
    del model_data
    gc.collect()

    preprocessing = np.load(run_dir.parent / "EXP-TABM-001" / "preprocessing.npz")
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
        validation_features,
        preprocessor,
        float(checkpoint["target_standard_deviation"]),
    )
    del validation_features, model
    torch.cuda.empty_cache()
    gc.collect()

    saved_mean = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tabm-001_valid.feather"
    )["prediction"].to_numpy(dtype=np.float32)
    recomputed_mean = members.mean(axis=1)
    mean_max_absolute_difference = float(
        np.max(np.abs(recomputed_mean - saved_mean))
    )
    mean_cosine_agreement = cosine(saved_mean, recomputed_mean)
    sorted_members = np.sort(members, axis=1)
    median = np.median(members, axis=1)
    aggregations = {
        "mean": recomputed_mean,
        "median": median,
        "trimmed_mean_1_each_side": sorted_members[:, 1:-1].mean(axis=1),
        "trimmed_mean_2_each_side": sorted_members[:, 2:-2].mean(axis=1),
        "trimmed_mean_4_each_side": sorted_members[:, 4:-4].mean(axis=1),
        "mean_median_75_25": 0.75 * recomputed_mean + 0.25 * median,
        "mean_median_50_50": 0.50 * recomputed_mean + 0.50 * median,
    }
    target = validation_rows["target"].to_numpy(dtype=np.float64)
    months = validation_rows["month"].to_numpy()
    aggregate_rows = [
        {"aggregation": name, **evaluate(target, prediction, months)}
        for name, prediction in aggregations.items()
    ]
    member_rows = [
        {"member": member, **evaluate(target, members[:, member], months)}
        for member in range(members.shape[1])
    ]
    aggregate_results = pd.DataFrame(aggregate_rows)
    member_results = pd.DataFrame(member_rows)
    aggregate_results.to_csv(run_dir / "aggregation_results.csv", index=False)
    member_results.to_csv(run_dir / "individual_member_results.csv", index=False)
    member_frame = validation_rows[["sample_id"]].copy()
    for member in range(members.shape[1]):
        member_frame[f"member_{member:02d}"] = members[:, member]
    member_frame.to_feather(run_dir / "member_predictions.feather")
    metadata = {
        "checkpoint": "outputs/models/exp-tabm-001.pt",
        "preprocessing": "data/interim/tree_experiments/EXP-TABM-001/preprocessing.npz",
        "validation_rows": int(len(validation_rows)),
        "member_count": int(members.shape[1]),
        "mean_reproduction": {
            "max_absolute_difference": mean_max_absolute_difference,
            "cosine_agreement": mean_cosine_agreement,
        },
        "aggregation_methods_use_labels": False,
        "aggregation_results": aggregate_results.to_dict(orient="records"),
        "individual_member_summary": {
            "overall_min": float(member_results["overall"].min()),
            "overall_max": float(member_results["overall"].max()),
            "recent_min": float(member_results["recent"].min()),
            "recent_max": float(member_results["recent"].max()),
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["mean_reproduction"], indent=2), flush=True)
    print(aggregate_results.to_string(index=False), flush=True)
    print(member_results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
