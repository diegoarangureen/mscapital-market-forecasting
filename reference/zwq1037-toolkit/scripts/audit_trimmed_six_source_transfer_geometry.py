"""Compare validation/test geometry for predeclared trimmed six-source blends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_trimmed_six_source_submissions import CANDIDATES


SOURCE_NAMES = ["original", "corrprune", "cosine", "xgboost", "lightgbm", "histgb"]
CENTERED_NAMES = {"original", "xgboost", "lightgbm"}


def transform(values: np.ndarray, centered: bool) -> np.ndarray:
    if centered:
        values = values - values.mean()
    norm = np.linalg.norm(values)
    if not norm:
        raise AssertionError("Source prediction vector has zero norm.")
    return values / norm


def matrix_geometry(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.corrcoef(matrix, rowvar=False), matrix.T @ matrix


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-BLEND-012-SIX-SOURCE-TRANSFER"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    validation_reference = pd.read_feather(
        prediction_dir / "exp-tabm-001-trim1_valid.feather"
    )
    validation_ids = validation_reference["sample_id"].to_numpy()
    cosine_valid = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-MEMBER-005-COMBINATION-ABLATION"
        / "cosine_aggregations_valid.feather"
    )
    validation_frames = {
        "original": validation_reference,
        "corrprune": pd.read_feather(
            prediction_dir / "exp-tabm-006-corrprune_valid.feather"
        ),
        "cosine": cosine_valid,
        "xgboost": pd.read_feather(prediction_dir / "exp-tree-053r_valid.feather"),
        "lightgbm": pd.read_feather(prediction_dir / "exp-tree-068_valid.feather"),
        "histgb": pd.read_feather(
            prediction_dir / "hist_gradient_boosting_target_centered_level2_valid.feather"
        ),
    }
    for name, frame in validation_frames.items():
        if not np.array_equal(frame["sample_id"].to_numpy(), validation_ids):
            raise AssertionError(f"Validation IDs are not aligned for {name}.")
    validation_columns = {
        "original": validation_frames["original"]["prediction"].to_numpy(dtype=np.float64),
        "corrprune": validation_frames["corrprune"]["prediction"].to_numpy(dtype=np.float64),
        "cosine": validation_frames["cosine"]["trim1_prediction"].to_numpy(dtype=np.float64),
        "xgboost": validation_frames["xgboost"]["prediction"].to_numpy(dtype=np.float64),
        "lightgbm": validation_frames["lightgbm"]["prediction"].to_numpy(dtype=np.float64),
        "histgb": validation_frames["histgb"]["prediction"].to_numpy(dtype=np.float64),
    }

    base_test = pd.read_feather(
        prediction_dir / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    original_test = pd.read_feather(
        prediction_dir / "tabm001_member_aggregations_fulltrain_test.feather"
    )
    corr_test = pd.read_feather(
        prediction_dir / "tabm006_corrprune_fulltrain_test.feather"
    )
    cosine_test = pd.read_feather(
        prediction_dir / "tabm003_cosine_member_aggregations_fulltrain_test.feather"
    )
    lgbm_test = pd.read_feather(prediction_dir / "lightgbm068_fulltrain_test.feather")
    hist_test = pd.read_feather(prediction_dir / "histgb007_fulltrain_test.feather")
    test_ids = base_test["sample_id"].to_numpy()
    for name, frame in [
        ("original", original_test),
        ("corrprune", corr_test),
        ("cosine", cosine_test),
        ("lightgbm", lgbm_test),
        ("histgb", hist_test),
    ]:
        if not np.array_equal(frame["sample_id"].to_numpy(), test_ids):
            raise AssertionError(f"Test IDs are not aligned for {name}.")
    test_columns = {
        "original": original_test["trim1_prediction"].to_numpy(dtype=np.float64),
        "corrprune": corr_test["prediction"].to_numpy(dtype=np.float64),
        "cosine": cosine_test["trim1_prediction"].to_numpy(dtype=np.float64),
        "xgboost": base_test["tree_prediction"].to_numpy(dtype=np.float64),
        "lightgbm": lgbm_test["prediction"].to_numpy(dtype=np.float64),
        "histgb": hist_test["prediction"].to_numpy(dtype=np.float64),
    }

    matrices = {}
    geometry = {}
    for split_name, columns in [("validation", validation_columns), ("test", test_columns)]:
        matrix = np.column_stack(
            [transform(columns[name], name in CENTERED_NAMES) for name in SOURCE_NAMES]
        )
        matrices[split_name] = matrix
        correlation, cosine = matrix_geometry(matrix)
        geometry[split_name] = {"correlation": correlation, "cosine": cosine}
        pd.DataFrame(correlation, index=SOURCE_NAMES, columns=SOURCE_NAMES).to_csv(
            run_dir / f"{split_name}_correlation.csv"
        )
        pd.DataFrame(cosine, index=SOURCE_NAMES, columns=SOURCE_NAMES).to_csv(
            run_dir / f"{split_name}_cosine.csv"
        )

    validation_public = pd.read_feather(prediction_dir / "exp-blend-001_valid.feather")[
        "prediction"
    ].to_numpy(dtype=np.float64)
    test_public = pd.read_csv(
        project_dir / "outputs" / "submissions" / "tabm001_tree053r_blend75_fulltrain.csv"
    )["prediction"].to_numpy(dtype=np.float64)
    rows = []
    for split_name, reference in [("validation", validation_public), ("test", test_public)]:
        matrix = matrices[split_name]
        reference_norm = np.linalg.norm(reference)
        for candidate_name, weight_map in CANDIDATES.items():
            weights = np.asarray([weight_map[name] for name in SOURCE_NAMES])
            candidate = matrix @ weights
            rows.append(
                {
                    "split": split_name,
                    "candidate": candidate_name,
                    "correlation_vs_public_best": float(np.corrcoef(candidate, reference)[0, 1]),
                    "relative_l2_change_vs_public_best": float(
                        np.linalg.norm(candidate - reference) / reference_norm
                    ),
                    "mean": float(candidate.mean()),
                    "std": float(candidate.std(ddof=0)),
                    "l2_norm": float(np.linalg.norm(candidate)),
                }
            )
    blend_geometry = pd.DataFrame(rows)
    blend_geometry.to_csv(run_dir / "blend_geometry.csv", index=False)

    correlation_delta = (
        geometry["test"]["correlation"] - geometry["validation"]["correlation"]
    )
    cosine_delta = geometry["test"]["cosine"] - geometry["validation"]["cosine"]
    result = {
        "source_names": SOURCE_NAMES,
        "validation_rows": int(len(validation_ids)),
        "test_rows": int(len(test_ids)),
        "pairwise_correlation_delta": {
            "mean_absolute": float(np.mean(np.abs(upper_values(correlation_delta)))),
            "max_absolute": float(np.max(np.abs(upper_values(correlation_delta)))),
        },
        "pairwise_cosine_delta": {
            "mean_absolute": float(np.mean(np.abs(upper_values(cosine_delta)))),
            "max_absolute": float(np.max(np.abs(upper_values(cosine_delta)))),
        },
        "blend_geometry": blend_geometry.to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
