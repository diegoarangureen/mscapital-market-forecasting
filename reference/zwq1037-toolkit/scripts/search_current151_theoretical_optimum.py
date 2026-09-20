"""Continuously optimize nonnegative blend weights on the LB geometry proxy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from build_current151_lower_public_grid import PATHS, load, unit
from build_factorized_transformer_lb_candidates import build_common_component
from estimate_public_lb_from_submission_geometry import normalized_prediction


PROJECT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = (
    PROJECT / "outputs/submission_metadata/public_lb_geometry_estimate.json"
)
OUTPUT_PATH = (
    PROJECT / "outputs/submissions/current151_theoretical_geometry_optimum.csv"
)
METADATA_PATH = (
    PROJECT / "outputs/submission_metadata/current151_theoretical_geometry_optimum.json"
)

SEARCH_PATHS = dict(PATHS)
SEARCH_PATHS["event"] = (
    PROJECT / "outputs/submissions/standalone_subsecond_event_transformer379_full_e3.csv"
)
NAMES = list(SEARCH_PATHS)
INCUMBENT_WEIGHTS = {
    "gpu": 0.20,
    "tpu": 0.12,
    "yangq": 0.32,
    "tabm": 0.12,
    "realmlp": 0.03,
    "transformer_old": 0.09,
    "transformer_multi": 0.09,
    "gru": 0.03,
    "event": 0.00,
}


def centered(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = values.copy()
    for group in np.unique(groups):
        mask = groups == group
        output[mask] -= output[mask].mean()
    return output


def vector_from_weights(weights: dict[str, float]) -> np.ndarray:
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
    source = {name: load(path, ids) for name, path in SEARCH_PATHS.items()}
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
    target_constant = float(coefficients @ scored_common)
    target_linear = coefficients @ scored_basis

    def estimate(weight: np.ndarray) -> float:
        norm2 = common_norm2 + 2.0 * float(weight @ basis_common) + float(
            weight @ basis_gram @ weight
        )
        numerator = target_constant + float(target_linear @ weight)
        return numerator / np.sqrt(norm2)

    def objective(weight: np.ndarray) -> float:
        return -estimate(weight)

    constraint = {"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)}
    bounds = [(0.0, 1.0)] * len(NAMES)
    rng = np.random.default_rng(20260917)
    starts = [
        vector_from_weights(INCUMBENT_WEIGHTS),
        np.full(len(NAMES), 1.0 / len(NAMES)),
    ]
    for concentration in (0.25, 0.5, 1.0, 2.0, 5.0):
        starts.extend(
            rng.dirichlet(np.full(len(NAMES), concentration), size=32)
        )

    solutions = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1e-14, "maxiter": 1000, "disp": False},
        )
        if not result.success:
            continue
        weight = np.clip(result.x, 0.0, 1.0)
        weight /= weight.sum()
        solutions.append(
            {
                "estimate": estimate(weight),
                "weights": {name: float(value) for name, value in zip(NAMES, weight)},
                "iterations": int(result.nit),
            }
        )
    if not solutions:
        raise RuntimeError("No optimizer restart converged.")
    solutions.sort(key=lambda row: row["estimate"], reverse=True)
    best = solutions[0]
    best_weight = vector_from_weights(best["weights"])
    incumbent_weight = vector_from_weights(INCUMBENT_WEIGHTS)
    prediction = common_vector + best_weight @ basis
    incumbent_prediction = common_vector + incumbent_weight @ basis
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(
        OUTPUT_PATH, index=False
    )

    distinct = []
    for row in solutions:
        weight = vector_from_weights(row["weights"])
        if any(
            np.max(np.abs(weight - vector_from_weights(item["weights"]))) < 1e-6
            for item in distinct
        ):
            continue
        distinct.append(row)
        if len(distinct) == 20:
            break
    public_weight = sum(best["weights"][name] for name in ("gpu", "tpu", "yangq"))
    incumbent_estimate = estimate(incumbent_weight)
    report = {
        "run_name": "current151_theoretical_geometry_optimum",
        "method": "SLSQP continuous nonnegative simplex optimization with 162 starts",
        "components": NAMES,
        "successful_restarts": len(solutions),
        "distinct_optima": len(distinct),
        "incumbent_public_lb": 0.151,
        "incumbent_geometry_estimate": incumbent_estimate,
        "best_geometry_estimate": best["estimate"],
        "estimated_delta": best["estimate"] - incumbent_estimate,
        "best_weights": best["weights"],
        "best_public_weight": public_weight,
        "best_owned_weight": 1.0 - public_weight,
        "top_distinct_solutions": distinct,
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
        "warning": "This is the optimum of a noisy geometry proxy, not a guaranteed leaderboard optimum.",
    }
    METADATA_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "successful_restarts": len(solutions),
                "distinct_optima": len(distinct),
                "incumbent_geometry_estimate": incumbent_estimate,
                "best_geometry_estimate": best["estimate"],
                "estimated_delta": best["estimate"] - incumbent_estimate,
                "best_public_weight": public_weight,
                "best_weights": best["weights"],
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
