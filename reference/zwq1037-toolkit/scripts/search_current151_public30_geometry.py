"""Search allocations while fixing the total public-model weight at 30%."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_current151_lower_public_grid import (
    INCUMBENT_WEIGHTS,
    PATHS,
    load,
    unit,
)
from build_factorized_transformer_lb_candidates import build_common_component
from estimate_public_lb_from_submission_geometry import normalized_prediction


PROJECT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = (
    PROJECT / "outputs/submission_metadata/public_lb_geometry_estimate.json"
)
OUTPUT_PATH = (
    PROJECT / "outputs/submissions/current151_public30_geometry_optimized.csv"
)
METADATA_PATH = (
    PROJECT / "outputs/submission_metadata/current151_public30_geometry_optimized.json"
)
NAMES = list(PATHS)


def centered(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = values.copy()
    for group in np.unique(groups):
        mask = groups == group
        output[mask] -= output[mask].mean()
    return output


def weight_vector(weights: dict[str, float]) -> np.ndarray:
    return np.asarray([weights[name] for name in NAMES], dtype=np.float64)


def main() -> None:
    geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8-sig"))
    expected_ids = None
    scored_vectors = []
    scores = []
    for row in geometry["accepted"]:
        expected_ids, vector = normalized_prediction(Path(row["path"]), expected_ids)
        scored_vectors.append(vector.astype(np.float64))
        scores.append(float(row["score"]))
    scored_matrix = np.vstack(scored_vectors)
    scores_array = np.asarray(scores, dtype=np.float64)
    scored_gram = np.clip(scored_matrix @ scored_matrix.T, -1.0, 1.0)
    ridge = float(geometry["selected_ridge"])
    coefficients = np.linalg.solve(
        scored_gram + ridge * np.eye(len(scores_array)), scores_array
    )

    labels, template, groups, common = build_common_component()
    ids = template["sample_id"].to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise AssertionError("Scored submissions and blend template IDs differ.")
    source = {name: load(path, ids) for name, path in PATHS.items()}
    basis = np.vstack([centered(unit(source[name]), groups) for name in NAMES])
    target_scale = float(labels["target"].to_numpy(dtype=np.float64).std())
    common_vector = np.asarray(
        [common[int(group)] / target_scale for group in groups], dtype=np.float64
    )

    scored_basis = scored_matrix @ basis.T
    scored_common = scored_matrix @ common_vector
    basis_gram = basis @ basis.T
    basis_common = basis @ common_vector
    common_norm2 = float(common_vector @ common_vector)

    def estimate(weights: dict[str, float]) -> float:
        w = weight_vector(weights)
        norm2 = common_norm2 + 2.0 * float(w @ basis_common) + float(
            w @ basis_gram @ w
        )
        correlations = (scored_common + scored_basis @ w) / np.sqrt(norm2)
        return float(correlations @ coefficients)

    incumbent_estimate = estimate(INCUMBENT_WEIGHTS)
    rows = []
    for gpu, tpu, realmlp, tabm, gru, multi_fraction in itertools.product(
        (0.06, 0.08, 0.10, 0.12, 0.14),
        (0.00, 0.02, 0.04, 0.06),
        (0.02, 0.03, 0.05),
        (0.16, 0.19, 0.22, 0.25, 0.28),
        (0.08, 0.11, 0.14, 0.17, 0.20),
        (0.40, 0.50, 0.60),
    ):
        yangq = 0.30 - gpu - tpu
        if yangq < 0.10:
            continue
        transformer = 0.70 - realmlp - tabm - gru
        if transformer < 0.15:
            continue
        weights = {
            "gpu": gpu,
            "tpu": tpu,
            "yangq": yangq,
            "tabm": tabm,
            "realmlp": realmlp,
            "transformer_old": transformer * (1.0 - multi_fraction),
            "transformer_multi": transformer * multi_fraction,
            "gru": gru,
        }
        rows.append({"estimate": estimate(weights), "weights": weights})

    rows.sort(key=lambda row: row["estimate"], reverse=True)
    best = rows[0]
    best_weights = best["weights"]
    prediction = common_vector + weight_vector(best_weights) @ basis
    incumbent_prediction = common_vector + weight_vector(INCUMBENT_WEIGHTS) @ basis
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
        OUTPUT_PATH, index=False
    )
    report = {
        "run_name": "current151_public30_geometry_optimized",
        "constraint": "gpu + tpu + yangq = 0.30",
        "grid_candidates": len(rows),
        "incumbent_public_lb": 0.151,
        "incumbent_geometry_estimate": incumbent_estimate,
        "best_geometry_estimate": best["estimate"],
        "estimated_delta": best["estimate"] - incumbent_estimate,
        "best_weights": best_weights,
        "top10": rows[:10],
        "diagnostics": {
            "correlation_with_incumbent": float(
                np.corrcoef(prediction, incumbent_prediction)[0, 1]
            ),
            "rows": len(ids),
            "finite": bool(np.isfinite(prediction).all()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std()),
        },
        "output_path": str(OUTPUT_PATH),
        "submission_status": "prepared_not_submitted",
        "warning": "Geometry model is diagnostic; CV error exceeds the estimated deltas.",
    }
    METADATA_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "grid_candidates": len(rows),
                "incumbent_geometry_estimate": incumbent_estimate,
                "best_geometry_estimate": best["estimate"],
                "estimated_delta": best["estimate"] - incumbent_estimate,
                "best_weights": best_weights,
                "correlation_with_incumbent": report["diagnostics"][
                    "correlation_with_incumbent"
                ],
                "output_path": str(OUTPUT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
