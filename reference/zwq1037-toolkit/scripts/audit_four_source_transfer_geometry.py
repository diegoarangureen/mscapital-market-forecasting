"""Compare four-source prediction geometry between validation and fulltrain test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


SOURCE_NAMES = [
    "tabm_original_centered",
    "tabm_corrprune_raw",
    "tabm_cosine_raw",
    "xgboost_centered",
]
AGGRESSIVE_WEIGHTS = np.asarray([0.45, 0.20, 0.20, 0.15], dtype=np.float64)
GUARDED_WEIGHTS = np.asarray([0.50, 0.25, 0.05, 0.20], dtype=np.float64)
PUBLIC_BEST_WEIGHTS = np.asarray([0.75, 0.00, 0.00, 0.25], dtype=np.float64)


def transformed(values: np.ndarray, centered: bool) -> np.ndarray:
    if centered:
        values = values - values.mean()
    norm = np.linalg.norm(values)
    if not norm:
        raise AssertionError("Prediction vector has zero norm.")
    return values / norm


def matrices(columns: list[np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = np.column_stack(columns)
    correlation = pd.DataFrame(
        np.corrcoef(matrix, rowvar=False), index=SOURCE_NAMES, columns=SOURCE_NAMES
    )
    cosine_gram = pd.DataFrame(
        matrix.T @ matrix, index=SOURCE_NAMES, columns=SOURCE_NAMES
    )
    return correlation, cosine_gram


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    indices = np.triu_indices(len(matrix), k=1)
    return matrix.to_numpy()[indices]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"

    validation_frames = {
        "tabm_original_centered": pd.read_feather(
            prediction_dir / "exp-tabm-001_valid.feather"
        ),
        "tabm_corrprune_raw": pd.read_feather(
            prediction_dir / "exp-tabm-006-corrprune_valid.feather"
        ),
        "tabm_cosine_raw": pd.read_feather(
            prediction_dir / "exp-tabm-003-cosine_valid.feather"
        ),
        "xgboost_centered": pd.read_feather(
            prediction_dir / "exp-tree-053r_valid.feather"
        ),
    }
    validation_reference = validation_frames["tabm_original_centered"]
    validation_rows = validation_reference[["sample_id", "month", "target"]]
    for frame in validation_frames.values():
        assert_prediction_alignment(frame, validation_rows)

    base_test = pd.read_feather(
        prediction_dir / "tabm001_tree053r_blend75_fulltrain_test.feather"
    )
    corrprune_test = pd.read_feather(
        prediction_dir / "tabm006_corrprune_fulltrain_test.feather"
    )
    cosine_test = pd.read_feather(
        prediction_dir / "tabm003_cosine_fulltrain_test.feather"
    )
    test_ids = base_test["sample_id"].to_numpy()
    for name, frame in [("corrprune", corrprune_test), ("cosine", cosine_test)]:
        if not np.array_equal(frame["sample_id"].to_numpy(), test_ids):
            raise AssertionError(f"{name} fulltrain test IDs are not aligned.")

    validation_raw = {
        name: frame["prediction"].to_numpy(dtype=np.float64)
        for name, frame in validation_frames.items()
    }
    test_raw = {
        "tabm_original_centered": base_test["tabm_prediction"].to_numpy(dtype=np.float64),
        "tabm_corrprune_raw": corrprune_test["prediction"].to_numpy(dtype=np.float64),
        "tabm_cosine_raw": cosine_test["prediction"].to_numpy(dtype=np.float64),
        "xgboost_centered": base_test["tree_prediction"].to_numpy(dtype=np.float64),
    }
    validation_columns = [
        transformed(validation_raw[name], name.endswith("_centered"))
        for name in SOURCE_NAMES
    ]
    test_columns = [
        transformed(test_raw[name], name.endswith("_centered")) for name in SOURCE_NAMES
    ]
    validation_correlation, validation_cosine = matrices(validation_columns)
    test_correlation, test_cosine = matrices(test_columns)
    correlation_delta = test_correlation - validation_correlation
    cosine_delta = test_cosine - validation_cosine

    validation_matrix = np.column_stack(validation_columns)
    test_matrix = np.column_stack(test_columns)
    blend_rows = []
    blend_vectors = {}
    for split_name, matrix in [("validation", validation_matrix), ("test", test_matrix)]:
        for blend_name, weights in [
            ("public_best_raw75_25", PUBLIC_BEST_WEIGHTS),
            ("aggressive_four", AGGRESSIVE_WEIGHTS),
            ("guarded_four", GUARDED_WEIGHTS),
        ]:
            vector = matrix @ weights
            blend_vectors[(split_name, blend_name)] = vector
            blend_rows.append(
                {
                    "split": split_name,
                    "blend": blend_name,
                    "mean": float(vector.mean()),
                    "std": float(vector.std(ddof=0)),
                    "l2_norm": float(np.linalg.norm(vector)),
                }
            )
    for split_name in ["validation", "test"]:
        reference = blend_vectors[(split_name, "public_best_raw75_25")]
        reference_norm = np.linalg.norm(reference)
        for blend_name in ["aggressive_four", "guarded_four"]:
            candidate = blend_vectors[(split_name, blend_name)]
            blend_rows.append(
                {
                    "split": split_name,
                    "blend": f"{blend_name}_vs_public_best",
                    "mean": float(np.corrcoef(candidate, reference)[0, 1]),
                    "std": float(np.linalg.norm(candidate - reference) / reference_norm),
                    "l2_norm": np.nan,
                }
            )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-AUDIT-TRANSFER-GEOMETRY"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_correlation": validation_correlation,
        "test_correlation": test_correlation,
        "correlation_delta": correlation_delta,
        "validation_cosine": validation_cosine,
        "test_cosine": test_cosine,
        "cosine_delta": cosine_delta,
    }
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv")
    blend_geometry = pd.DataFrame(blend_rows)
    blend_geometry.to_csv(run_dir / "blend_geometry.csv", index=False)
    correlation_pair_delta = upper_triangle_values(correlation_delta)
    cosine_pair_delta = upper_triangle_values(cosine_delta)
    result = {
        "source_names": SOURCE_NAMES,
        "validation_rows": int(len(validation_matrix)),
        "test_rows": int(len(test_matrix)),
        "pairwise_correlation_delta": {
            "mean_absolute": float(np.mean(np.abs(correlation_pair_delta))),
            "max_absolute": float(np.max(np.abs(correlation_pair_delta))),
        },
        "pairwise_cosine_delta": {
            "mean_absolute": float(np.mean(np.abs(cosine_pair_delta))),
            "max_absolute": float(np.max(np.abs(cosine_pair_delta))),
        },
        "blend_geometry": blend_geometry.to_dict(orient="records"),
        "interpretation_rule": (
            "smaller pairwise geometry shifts support transfer of validation blend weights"
        ),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("validation correlation")
    print(validation_correlation.to_string())
    print("test correlation")
    print(test_correlation.to_string())
    print("correlation delta")
    print(correlation_delta.to_string())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
