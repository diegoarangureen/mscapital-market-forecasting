"""Audit label-free member aggregations for cosine and corr-pruned TabM."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_tabm_member_aggregation import cosine, evaluate, predict_members
from exp_tabm_001_exp053r_features import (
    BatchPreprocessor,
    load_exp053r_data,
    make_model,
)


EXPERIMENT_ID = "EXP-TABM-MEMBER-004-VARIANTS"
VARIANTS = {
    "tabm003_cosine": {
        "checkpoint": "outputs/models/exp-tabm-003-cosine.pt",
        "preprocessing": (
            "data/interim/tree_experiments/EXP-TABM-003-COSINE/preprocessing.npz"
        ),
        "saved_prediction": "outputs/predictions/exp-tabm-003-cosine_valid.feather",
        "feature_mode": "all",
    },
    "tabm006_corrprune": {
        "checkpoint": "outputs/models/exp-tabm-006-corrprune.pt",
        "preprocessing": (
            "data/interim/tree_experiments/EXP-TABM-006-CORRPRUNE/preprocessing.npz"
        ),
        "saved_prediction": (
            "outputs/predictions/exp-tabm-006-corrprune_valid.feather"
        ),
        "feature_mode": "corrprune",
    },
}


def aggregate_members(members: np.ndarray) -> dict[str, np.ndarray]:
    sorted_members = np.sort(members, axis=1)
    median = np.median(members, axis=1)
    mean = members.mean(axis=1)
    return {
        "mean": mean,
        "median": median,
        "trimmed_mean_1_each_side": sorted_members[:, 1:-1].mean(axis=1),
        "trimmed_mean_2_each_side": sorted_members[:, 2:-2].mean(axis=1),
        "mean_median_75_25": 0.75 * mean + 0.25 * median,
    }


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

    model_data, all_features, _, _ = load_exp053r_data(project_dir)
    selection = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "exp053r_correlated_feature_selection.json"
        ).read_text(encoding="utf-8")
    )
    if selection["selection_source"] != "training months 0-59 only":
        raise AssertionError("Correlation pruning selection is not training-only.")
    kept = set(selection["kept_features_in_gain_order"])
    corrpruned_features = [name for name in all_features if name in kept]
    if len(corrpruned_features) != 293:
        raise AssertionError(
            f"Expected 293 corr-pruned features, got {len(corrpruned_features)}."
        )

    validation_mask = model_data["month"].to_numpy() >= 60
    validation_rows = model_data.loc[
        validation_mask, ["sample_id", "month", "target"]
    ].reset_index(drop=True)
    target = validation_rows["target"].to_numpy(dtype=np.float64)
    months = validation_rows["month"].to_numpy()
    results: list[dict[str, object]] = []
    reproductions: dict[str, object] = {}

    for variant_name, specification in VARIANTS.items():
        features = (
            all_features
            if specification["feature_mode"] == "all"
            else corrpruned_features
        )
        checkpoint = torch.load(
            project_dir / specification["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        if features != checkpoint["feature_columns"]:
            raise AssertionError(f"Feature schema mismatch for {variant_name}.")
        validation_features = model_data.loc[
            validation_mask, features
        ].to_numpy(dtype=np.float32, copy=True)
        preprocessing = np.load(project_dir / specification["preprocessing"])
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
        aggregations = aggregate_members(members)
        saved = pd.read_feather(
            project_dir / specification["saved_prediction"]
        )["prediction"].to_numpy(dtype=np.float32)
        recomputed = aggregations["mean"]
        reproductions[variant_name] = {
            "max_absolute_difference": float(np.max(np.abs(recomputed - saved))),
            "cosine_agreement": cosine(saved, recomputed),
            "member_count": int(members.shape[1]),
        }
        for aggregation_name, prediction in aggregations.items():
            results.append(
                {
                    "variant": variant_name,
                    "aggregation": aggregation_name,
                    **evaluate(target, prediction, months),
                }
            )
        del validation_features, checkpoint, model, members, aggregations
        torch.cuda.empty_cache()
        gc.collect()

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(run_dir / "aggregation_results.csv", index=False)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "validation_rows": int(len(validation_rows)),
        "validation_months": sorted(int(value) for value in np.unique(months)),
        "aggregation_methods_use_labels": False,
        "reproductions": reproductions,
        "results": result_frame.to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(reproductions, indent=2), flush=True)
    print(result_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
