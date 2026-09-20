"""Test all mean/trim1 choices for three TabM sources in leading blends."""

from __future__ import annotations

import gc
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_tabm_member_aggregation import predict_members
from audit_trimmed_tabm_in_blends import centered_unit, evaluate, unit
from exp_tabm_001_exp053r_features import (
    BatchPreprocessor,
    load_exp053r_data,
    make_model,
)
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


EXPERIMENT_ID = "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
VARIANTS = {
    "cosine": {
        "checkpoint": "outputs/models/exp-tabm-003-cosine.pt",
        "preprocessing": (
            "data/interim/tree_experiments/EXP-TABM-003-COSINE/preprocessing.npz"
        ),
        "saved_prediction": "outputs/predictions/exp-tabm-003-cosine_valid.feather",
        "feature_mode": "all",
    },
    "corrprune": {
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
BLENDS = {
    "aggressive": {
        "original": 0.45,
        "corrprune": 0.20,
        "cosine": 0.20,
        "xgboost": 0.15,
    },
    "guarded": {
        "original": 0.50,
        "corrprune": 0.25,
        "cosine": 0.05,
        "xgboost": 0.20,
    },
}


def mean_and_trim1(members: np.ndarray) -> dict[str, np.ndarray]:
    sorted_members = np.sort(members, axis=1)
    return {
        "mean": members.mean(axis=1),
        "trim1": sorted_members[:, 1:-1].mean(axis=1),
    }


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")

    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_feather(prediction_dir / "exp-tabm-001_valid.feather")
    validation_rows = reference[["sample_id", "month", "target"]]
    original_member_frame = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-001"
        / "member_predictions.feather"
    )
    if not np.array_equal(
        reference["sample_id"].to_numpy(),
        original_member_frame["sample_id"].to_numpy(),
    ):
        raise AssertionError("Original TabM member predictions are not aligned.")
    member_columns = [
        column for column in original_member_frame if column.startswith("member_")
    ]
    source_predictions: dict[str, dict[str, np.ndarray]] = {
        "original": mean_and_trim1(
            original_member_frame[member_columns].to_numpy(dtype=np.float32)
        )
    }
    del original_member_frame
    gc.collect()

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
    validation_mask = model_data["month"].to_numpy() >= 60

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
        if checkpoint["feature_columns"] != features:
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
        source_predictions[variant_name] = mean_and_trim1(members)
        saved = pd.read_feather(project_dir / specification["saved_prediction"])
        assert_prediction_alignment(saved, validation_rows)
        mean_difference = np.abs(
            source_predictions[variant_name]["mean"]
            - saved["prediction"].to_numpy(dtype=np.float32)
        )
        reproductions[variant_name] = {
            "max_absolute_difference": float(mean_difference.max()),
            "member_count": int(members.shape[1]),
        }
        output = validation_rows.copy()
        output["mean_prediction"] = source_predictions[variant_name]["mean"]
        output["trim1_prediction"] = source_predictions[variant_name]["trim1"]
        output.to_feather(run_dir / f"{variant_name}_aggregations_valid.feather")
        del validation_features, preprocessing, preprocessor, model, members, checkpoint
        torch.cuda.empty_cache()
        gc.collect()
    del model_data
    gc.collect()

    xgboost = pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather")
    assert_prediction_alignment(xgboost, validation_rows)
    xgboost_unit = centered_unit(xgboost["prediction"].to_numpy(dtype=np.float64))
    transformed_sources = {
        "original": {
            aggregation: centered_unit(prediction.astype(np.float64))
            for aggregation, prediction in source_predictions["original"].items()
        },
        "corrprune": {
            aggregation: unit(prediction.astype(np.float64))
            for aggregation, prediction in source_predictions["corrprune"].items()
        },
        "cosine": {
            aggregation: unit(prediction.astype(np.float64))
            for aggregation, prediction in source_predictions["cosine"].items()
        },
    }
    target = validation_rows["target"].to_numpy(dtype=np.float64)
    months = validation_rows["month"].to_numpy()
    result_rows: list[dict[str, object]] = []

    for blend_name, weights in BLENDS.items():
        baseline = (
            weights["original"] * transformed_sources["original"]["mean"]
            + weights["corrprune"] * transformed_sources["corrprune"]["mean"]
            + weights["cosine"] * transformed_sources["cosine"]["mean"]
            + weights["xgboost"] * xgboost_unit
        )
        for choices in itertools.product(["mean", "trim1"], repeat=3):
            choice = dict(zip(["original", "corrprune", "cosine"], choices))
            prediction = (
                weights["original"] * transformed_sources["original"][choice["original"]]
                + weights["corrprune"]
                * transformed_sources["corrprune"][choice["corrprune"]]
                + weights["cosine"] * transformed_sources["cosine"][choice["cosine"]]
                + weights["xgboost"] * xgboost_unit
            )
            metrics = evaluate(target, prediction, months, baseline)
            result_rows.append(
                {
                    "blend": blend_name,
                    "original": choice["original"],
                    "corrprune": choice["corrprune"],
                    "cosine": choice["cosine"],
                    **metrics,
                    "all_primary_months_non_decreasing": (
                        metrics["primary_month_min_change"] >= 0.0
                    ),
                    "all_primary_lomo_non_decreasing": (
                        metrics["primary_lomo_min"] >= 0.0
                    ),
                }
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(run_dir / "combination_results.csv", index=False)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "aggregation_uses_labels": False,
        "reproductions": reproductions,
        "weights": BLENDS,
        "results": results.to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(reproductions, indent=2), flush=True)
    print(
        results.sort_values(
            ["blend", "no66", "recent"], ascending=[True, False, False]
        ).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
