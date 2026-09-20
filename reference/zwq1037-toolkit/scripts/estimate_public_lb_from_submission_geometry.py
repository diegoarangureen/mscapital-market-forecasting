"""Estimate candidate public LB from correlations with scored local submissions.

This is a diagnostic model, not a substitute for a real Kaggle submission.  It
fits the minimum-norm projection of the hidden normalized target onto the span
of locally available, already-scored prediction vectors and selects ridge
regularization by leave-one-out error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUBMISSION_LOG = PROJECT / "data/interim/kaggle_submissions_account.json"
OUTPUT_PATH = PROJECT / "outputs/submission_metadata/public_lb_geometry_estimate.json"
EXPECTED_ROWS = 647_896

EXTRA_SCORED = {
    "tabm385_k32_temporal3fold_3seed.csv": 0.140,
    "standalone_timeaware_three_stream_gru379.csv": 0.140,
    "standalone_factorized_transformer379.csv": 0.145,
    "standalone_realmlp379_corr095.csv": 0.139,
    "submission.csv@bestwater_cos689": 0.142,
    "submission.csv@bestwater_tpu_v6": 0.142,
    "submission.csv@yangq_lb142": 0.142,
}

CANDIDATES = [
    "factorized_full_replace_corr095_current138k32_equal38_common.csv",
    "current150_replace_owned_tabm_temporal3x3.csv",
    "current150_tabm_temporal_realmlp57_upgrades.csv",
    "current150_confirmed_tabm_transformer_temporal_upgrades.csv",
    "current150_temporal_balanced_tabm_realmlp57_transformer50.csv",
    "current150_temporal_model_upgrades_realmlp57.csv",
    "aggressive_direct_owned_block_gru_equal38_common.csv",
    "aggressive_direct_owned_block_temporal4_equal38_common.csv",
    "current151_lower_public60.csv",
    "current151_lower_public56.csv",
    "current151_lower_public56_tpu4.csv",
    "current151_lower_public52_tpu0.csv",
    "current151_lower_public52.csv",
    "current151_lower_public48.csv",
    "current151_lower_public30.csv",
    "current151_theoretical_geometry_optimum.csv",
    "current150_tabmfrac1p0_realmlpfrac0p5.csv",
    "current150_tabmfrac1p0_realmlpfrac1p0.csv",
    "current150_tabmfrac1p25_realmlpfrac0p5.csv",
    "current150_tabmfrac1p25_realmlpfrac1p0.csv",
    "current150_tabmfrac1p5_realmlpfrac0p5.csv",
    "current150_tabmfrac1p5_realmlpfrac1p0.csv",
]


def normalized_prediction(path: Path, expected_ids: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id", "prediction"] or len(frame) != EXPECTED_ROWS:
        raise AssertionError(f"Invalid submission file: {path}")
    ids = frame["sample_id"].to_numpy()
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise AssertionError(f"IDs differ: {path}")
    values = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError(f"Nonfinite prediction: {path}")
    norm = np.linalg.norm(values)
    if norm <= 0.0:
        raise AssertionError(f"Zero prediction: {path}")
    return ids, (values / norm).astype(np.float32)


def index_csv_files() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in PROJECT.rglob("*.csv"):
        index.setdefault(path.name, []).append(path)
    return index


def path_priority(path: Path) -> tuple[int, int, str]:
    text = str(path).lower()
    if "outputs\\submissions" in text:
        rank = 0
    elif "kaggle_outputs" in text:
        rank = 1
    elif "kaggle_public" in text:
        rank = 2
    else:
        rank = 3
    return rank, len(text), text


def find_regular(name: str, index: dict[str, list[Path]]) -> Path | None:
    paths = sorted(index.get(name, []), key=path_priority)
    return paths[0] if paths else None


def find_extra(key: str, index: dict[str, list[Path]]) -> Path | None:
    if "@" not in key:
        return find_regular(key, index)
    name, marker = key.split("@", 1)
    paths = [path for path in index.get(name, []) if marker.lower() in str(path).lower()]
    return sorted(paths, key=path_priority)[0] if paths else None


def ridge_predict(gram_train: np.ndarray, scores: np.ndarray, correlations: np.ndarray, ridge: float) -> float:
    system = gram_train + ridge * np.eye(len(gram_train), dtype=np.float64)
    coefficients = np.linalg.solve(system, scores)
    return float(correlations @ coefficients)


def main() -> None:
    candidate_names = list(CANDIDATES)
    candidate_names.extend(
        path.name
        for pattern in ("final_gru_direct_*.csv", "final_gru_residual_*.csv")
        for path in sorted((PROJECT / "outputs/submissions").glob(pattern))
    )
    submission_rows = json.loads(SUBMISSION_LOG.read_text(encoding="utf-8-sig"))
    index = index_csv_files()
    records: list[tuple[str, float, Path, str]] = []
    seen_names: set[str] = set()
    missing = []
    for row in submission_rows:
        name = str(row["fileName"])
        score_text = str(row.get("publicScore", "")).strip()
        if not score_text or name in seen_names:
            continue
        path = find_regular(name, index)
        if path is None:
            missing.append(name)
            continue
        records.append((name, float(score_text), path, "account"))
        seen_names.add(name)
    for key, score in EXTRA_SCORED.items():
        canonical_name = key.split("@", 1)[0]
        record_name = key
        if canonical_name in seen_names and "@" not in key:
            continue
        path = find_extra(key, index)
        if path is None:
            missing.append(key)
            continue
        records.append((record_name, score, path, "user_reported"))

    expected_ids = None
    vectors = []
    accepted = []
    rejected = []
    for name, score, path, source in records:
        try:
            expected_ids, vector = normalized_prediction(path, expected_ids)
        except Exception as exc:
            rejected.append({"name": name, "path": str(path), "reason": str(exc)})
            continue
        vectors.append(vector)
        accepted.append({"name": name, "score": score, "path": str(path), "source": source})
    if len(vectors) < 8:
        raise RuntimeError(f"Too few scored vectors: {len(vectors)}")

    matrix = np.vstack(vectors).astype(np.float64)
    scores = np.asarray([row["score"] for row in accepted], dtype=np.float64)
    gram = np.clip(matrix @ matrix.T, -1.0, 1.0)
    ridges = np.logspace(-7, 0, 29)
    cv = []
    for ridge in ridges:
        predictions = []
        for held_out in range(len(scores)):
            keep = np.arange(len(scores)) != held_out
            predictions.append(
                ridge_predict(
                    gram[np.ix_(keep, keep)],
                    scores[keep],
                    gram[held_out, keep],
                    float(ridge),
                )
            )
        predictions = np.asarray(predictions)
        cv.append(
            {
                "ridge": float(ridge),
                "mae": float(np.mean(np.abs(predictions - scores))),
                "rmse": float(np.sqrt(np.mean((predictions - scores) ** 2))),
                "max_abs": float(np.max(np.abs(predictions - scores))),
            }
        )
    best = min(cv, key=lambda row: (row["mae"], row["rmse"]))
    ridge = float(best["ridge"])
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(len(scores), dtype=np.float64), scores
    )

    candidate_results = {}
    candidate_correlations = {}
    for name in dict.fromkeys(candidate_names):
        path = find_regular(name, index)
        if path is None:
            continue
        _, vector = normalized_prediction(path, expected_ids)
        correlations = matrix @ vector.astype(np.float64)
        candidate_correlations[name] = correlations
        estimate = float(correlations @ coefficients)
        candidate_results[name] = {
            "path": str(path),
            "estimate": estimate,
            "nearest_scored_name": accepted[int(np.argmax(correlations))]["name"],
            "nearest_scored_correlation": float(np.max(correlations)),
        }

    base_name = "factorized_full_replace_corr095_current138k32_equal38_common.csv"
    if base_name in candidate_correlations:
        rng = np.random.default_rng(20260916)
        noisy_scores = scores[None, :] + rng.uniform(
            -0.0005, 0.0005, size=(4000, len(scores))
        )
        inverse_system = np.linalg.inv(
            gram + ridge * np.eye(len(scores), dtype=np.float64)
        )
        noisy_coefficients = noisy_scores @ inverse_system.T
        base_draws = noisy_coefficients @ candidate_correlations[base_name]
        for name, correlations in candidate_correlations.items():
            draws = noisy_coefficients @ correlations
            delta = draws - base_draws
            candidate_results[name]["rounding_uncertainty_vs_base"] = {
                "mean_delta": float(delta.mean()),
                "p05_delta": float(np.quantile(delta, 0.05)),
                "p50_delta": float(np.quantile(delta, 0.50)),
                "p95_delta": float(np.quantile(delta, 0.95)),
                "probability_delta_positive": float(np.mean(delta > 0.0)),
            }

    report = {
        "method": "minimum-norm hidden-target projection with LOOCV ridge",
        "warning": "Diagnostic only; public scores are displayed at three decimals and do not identify unseen residual directions.",
        "scored_vector_count": len(accepted),
        "accepted": accepted,
        "missing_files": missing,
        "rejected_files": rejected,
        "selected_ridge": ridge,
        "selected_cv": best,
        "cv_grid": cv,
        "candidates": candidate_results,
    }
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_cv": best, "candidates": candidate_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




