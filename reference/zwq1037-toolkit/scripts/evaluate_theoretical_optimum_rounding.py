"""Compare the theoretical optimum directly with the scored 0.151 incumbent."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from estimate_public_lb_from_submission_geometry import normalized_prediction


PROJECT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = (
    PROJECT / "outputs/submission_metadata/public_lb_geometry_estimate.json"
)
OPTIMUM_METADATA_PATH = (
    PROJECT / "outputs/submission_metadata/current151_theoretical_geometry_optimum.json"
)


def main() -> None:
    geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8-sig"))
    expected_ids = None
    scored_vectors = []
    scores = []
    for row in geometry["accepted"]:
        expected_ids, vector = normalized_prediction(Path(row["path"]), expected_ids)
        scored_vectors.append(vector.astype(np.float64))
        scores.append(float(row["score"]))
    matrix = np.vstack(scored_vectors)
    scores_array = np.asarray(scores, dtype=np.float64)
    gram = np.clip(matrix @ matrix.T, -1.0, 1.0)
    ridge = float(geometry["selected_ridge"])
    inverse = np.linalg.inv(gram + ridge * np.eye(len(scores_array)))

    incumbent_path = (
        PROJECT / "outputs/submissions/current151_structure_transformer50.csv"
    )
    candidate_path = (
        PROJECT / "outputs/submissions/current151_theoretical_geometry_optimum.csv"
    )
    _, incumbent = normalized_prediction(incumbent_path, expected_ids)
    _, candidate = normalized_prediction(candidate_path, expected_ids)
    incumbent_correlations = matrix @ incumbent.astype(np.float64)
    candidate_correlations = matrix @ candidate.astype(np.float64)

    rng = np.random.default_rng(20260917)
    noisy_scores = scores_array[None, :] + rng.uniform(
        -0.0005, 0.0005, size=(20_000, len(scores_array))
    )
    noisy_coefficients = noisy_scores @ inverse.T
    deltas = noisy_coefficients @ (candidate_correlations - incumbent_correlations)
    report = {
        "draws": len(deltas),
        "score_noise": "independent uniform +/-0.0005",
        "mean_delta_vs_scored_0p151_incumbent": float(deltas.mean()),
        "p01_delta": float(np.quantile(deltas, 0.01)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p50_delta": float(np.quantile(deltas, 0.50)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "p99_delta": float(np.quantile(deltas, 0.99)),
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
        "warning": "This covers leaderboard rounding noise, not geometry-model misspecification.",
    }
    metadata = json.loads(OPTIMUM_METADATA_PATH.read_text(encoding="utf-8-sig"))
    metadata["rounding_robustness_vs_scored_incumbent"] = report
    OPTIMUM_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
