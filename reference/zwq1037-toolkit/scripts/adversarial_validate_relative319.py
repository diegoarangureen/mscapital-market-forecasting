"""Minimal adversarial validation and test-likeness OOF diagnostics for rel319."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_014_quantile_stage_curves import unit
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from export_full_tabm001_member_aggregations import load_test_only
from train_full_quantile_tabm_relative319_submission import add_split_relative_features


EXPERIMENT_ID = "EXP-AUDIT-003-ADVERSARIAL-REL319"
SEED = 42
ACTUAL_SAMPLE_PER_SOURCE = 250_000
PSEUDO_SAMPLE_PER_SOURCE = 150_000
BLOCK_SIZE = 2048


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.linalg.norm(target) * np.linalg.norm(prediction))
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def sample_indices(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(indices) <= count:
        return indices.copy()
    return np.random.default_rng(seed).choice(indices, size=count, replace=False)


def validation_blocks(sample_ids: np.ndarray, domains: np.ndarray) -> np.ndarray:
    blocks = sample_ids.astype(np.int64) // BLOCK_SIZE
    keys = blocks * 2 + domains.astype(np.int64)
    hashed = (keys * 1_103_515_245 + 12_345) & 0x7FFFFFFF
    return (hashed % 5) == 0


def make_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=100.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=255,
        random_state=SEED,
        n_jobs=2,
        device="cuda",
        importance_type="gain",
        verbosity=0,
    )


def fit_source_classifier(
    *,
    name: str,
    source_zero: np.ndarray,
    source_zero_ids: np.ndarray,
    source_one: np.ndarray,
    source_one_ids: np.ndarray,
    feature_columns: list[str],
    run_dir: Path,
) -> tuple[XGBClassifier, dict, pd.DataFrame]:
    features = np.concatenate([source_zero, source_one], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(source_zero), dtype=np.int8),
            np.ones(len(source_one), dtype=np.int8),
        ]
    )
    ids = np.concatenate([source_zero_ids, source_one_ids])
    valid_mask = validation_blocks(ids, labels)
    if min(np.bincount(labels[valid_mask], minlength=2)) == 0:
        raise AssertionError(f"{name} validation lacks a source class.")
    model = make_classifier()
    started = time.perf_counter()
    model.fit(features[~valid_mask], labels[~valid_mask])
    seconds = time.perf_counter() - started
    probability = model.predict_proba(features[valid_mask])[:, 1]
    auc = float(roc_auc_score(labels[valid_mask], probability))
    importance = pd.DataFrame(
        {
            "experiment": name,
            "feature": feature_columns,
            "gain_importance": model.feature_importances_.astype(np.float64),
        }
    ).sort_values("gain_importance", ascending=False, ignore_index=True)
    importance.insert(1, "rank", np.arange(1, len(importance) + 1))
    metadata = {
        "name": name,
        "source_zero_rows": int(len(source_zero)),
        "source_one_rows": int(len(source_one)),
        "fit_rows": int((~valid_mask).sum()),
        "validation_rows": int(valid_mask.sum()),
        "validation_source_zero_rows": int(np.sum(labels[valid_mask] == 0)),
        "validation_source_one_rows": int(np.sum(labels[valid_mask] == 1)),
        "group_block_size": BLOCK_SIZE,
        "validation_auc": auc,
        "training_seconds": seconds,
    }
    save_model_safely(model, run_dir / f"{name}_classifier.json")
    print(f"{name}: auc={auc:.6f}, seconds={seconds:.1f}", flush=True)
    del features, labels, ids, valid_mask, probability
    gc.collect()
    return model, metadata, importance


def sampled_distribution(
    name: str,
    values: np.ndarray,
    feature_columns: list[str],
    top_features: list[str],
) -> list[dict[str, object]]:
    rows = []
    positions = {feature: feature_columns.index(feature) for feature in top_features}
    for feature, position in positions.items():
        column = values[:, position].astype(np.float64, copy=False)
        finite = column[np.isfinite(column)]
        quantiles = np.quantile(finite, [0.10, 0.50, 0.90]) if finite.size else [np.nan] * 3
        rows.append(
            {
                "source": name,
                "feature": feature,
                "rows": int(len(column)),
                "missing_rate": float(1.0 - finite.size / len(column)),
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std(ddof=0)) if finite.size else None,
                "q10": float(quantiles[0]),
                "q50": float(quantiles[1]),
                "q90": float(quantiles[2]),
            }
        )
    return rows


def load_legacy_static(
    project_dir: Path, fold_name: str, validation_ids: np.ndarray
) -> np.ndarray:
    if fold_name == "train049_valid5059":
        frame = pd.read_feather(
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-BACKTEST-001-B001-TRAIN049-VALID5059"
            / "validation_predictions.feather"
        )
        if not np.array_equal(frame["sample_id"].to_numpy(), validation_ids):
            raise AssertionError("Legacy 50-59 static prediction IDs differ.")
        return frame["b001_trim1"].to_numpy(dtype=np.float64)

    tabm_frame = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tabm-001-trim1_valid.feather"
    )
    tree_frame = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
    )
    if not np.array_equal(tabm_frame["sample_id"].to_numpy(), validation_ids):
        raise AssertionError("Legacy TabM 60-70 IDs differ.")
    if not np.array_equal(tree_frame["sample_id"].to_numpy(), validation_ids):
        raise AssertionError("Legacy tree 60-70 IDs differ.")
    return 0.75 * unit(tabm_frame["prediction"].to_numpy(dtype=np.float64)) + 0.25 * unit(
        tree_frame["prediction"].to_numpy(dtype=np.float64)
    )


def stratified_oof_report(
    project_dir: Path,
    fold_name: str,
    late_ids: np.ndarray,
    late_months: np.ndarray,
    late_probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_name == "train049_valid5059":
        start, end = 50, 59
    else:
        start, end = 60, 70
    fold_mask = (late_months >= start) & (late_months <= end)
    ids = late_ids[fold_mask]
    probabilities = late_probabilities[fold_mask]
    prediction_frame = pd.read_feather(
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-TABM-016-RELATIVE-SCALE-ADD12"
        / fold_name
        / "validation_predictions.feather"
    )
    if not np.array_equal(prediction_frame["sample_id"].to_numpy(), ids):
        raise AssertionError(f"Domain scores are misaligned for {fold_name}.")
    target = prediction_frame["target"].to_numpy(dtype=np.float64)
    months = prediction_frame["month"].to_numpy()
    legacy = load_legacy_static(project_dir, fold_name, ids)
    current = prediction_frame["b001_trim1"].to_numpy(dtype=np.float64)
    tree = prediction_frame["xgboost"].to_numpy(dtype=np.float64)
    predictions = {
        "legacy_static": legacy,
        "relative319_static": current,
        "xgboost053r": tree,
    }
    if fold_name == "train059_valid6070":
        gru = pd.read_feather(
            project_dir / "outputs" / "predictions" / "exp-gru-001_valid.feather"
        )
        if not np.array_equal(gru["sample_id"].to_numpy(), ids):
            raise AssertionError("GRU-001 IDs differ.")
        gru_prediction = gru["prediction"].to_numpy(dtype=np.float64)
        predictions["gru001"] = gru_prediction
        predictions["relative319_plus_gru10"] = 0.90 * unit(current) + 0.10 * unit(
            gru_prediction
        )

    boundaries = np.quantile(probabilities, [1.0 / 3.0, 2.0 / 3.0])
    tiers = np.where(
        probabilities <= boundaries[0],
        "low",
        np.where(probabilities <= boundaries[1], "middle", "high"),
    )
    score_rows = []
    composition_rows = []
    for tier in ("low", "middle", "high"):
        tier_mask = tiers == tier
        score_rows.append(
            {
                "fold": fold_name,
                "tier": tier,
                "rows": int(tier_mask.sum()),
                "probability_mean": float(probabilities[tier_mask].mean()),
                **{
                    model_name: cosine(target[tier_mask], prediction[tier_mask])
                    for model_name, prediction in predictions.items()
                },
            }
        )
        for month in range(start, end + 1):
            composition_rows.append(
                {
                    "fold": fold_name,
                    "tier": tier,
                    "month": month,
                    "rows": int(np.sum(tier_mask & (months == month))),
                    "share_within_tier": float(
                        np.sum(tier_mask & (months == month)) / tier_mask.sum()
                    ),
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(composition_rows)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data, baseline_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, train_data)
    feature_columns = [*baseline_columns, *RELATIVE_COLUMNS]
    if len(feature_columns) != 319 or len(set(feature_columns)) != 319:
        raise AssertionError("Expected 319 unique features.")
    months = train_data["month"].to_numpy(dtype=np.int16)
    train_ids = train_data["sample_id"].to_numpy(dtype=np.int32)
    generator = np.random.default_rng(SEED)

    experiment_results = {}
    importance_frames = []
    pseudo_specs = {
        "pseudo_0_39_vs_50_59": (
            np.flatnonzero(months <= 39),
            np.flatnonzero((months >= 50) & (months <= 59)),
        ),
        "pseudo_0_49_vs_60_70": (
            np.flatnonzero(months <= 49),
            np.flatnonzero((months >= 60) & (months <= 70)),
        ),
    }
    for offset, (name, (zero_pool, one_pool)) in enumerate(pseudo_specs.items()):
        zero_indices = sample_indices(zero_pool, PSEUDO_SAMPLE_PER_SOURCE, SEED + 10 + offset)
        one_indices = sample_indices(one_pool, PSEUDO_SAMPLE_PER_SOURCE, SEED + 20 + offset)
        zero_features = train_data.iloc[zero_indices][feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        one_features = train_data.iloc[one_indices][feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        pseudo_model, metadata, importance = fit_source_classifier(
            name=name,
            source_zero=zero_features,
            source_zero_ids=train_ids[zero_indices],
            source_one=one_features,
            source_one_ids=train_ids[one_indices],
            feature_columns=feature_columns,
            run_dir=run_dir,
        )
        experiment_results[name] = metadata
        importance_frames.append(importance)
        del pseudo_model, zero_features, one_features
        gc.collect()

    actual_train_pool = np.flatnonzero(months <= 49)
    actual_train_indices = sample_indices(
        actual_train_pool, ACTUAL_SAMPLE_PER_SOURCE, SEED + 30
    )
    actual_train_features = train_data.iloc[actual_train_indices][feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )
    late_mask = months >= 50
    late_ids = train_ids[late_mask]
    late_months = months[late_mask]
    late_features = train_data.loc[late_mask, feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )
    actual_train_ids = train_ids[actual_train_indices]
    del train_data
    gc.collect()

    test_data, template, test_columns = load_test_only(project_dir)
    add_split_relative_features(project_dir, test_data, "test")
    test_feature_columns = [*test_columns, *RELATIVE_COLUMNS]
    if test_feature_columns != feature_columns:
        raise AssertionError("Train and test feature schemas differ.")
    test_indices = sample_indices(
        np.arange(len(test_data)), ACTUAL_SAMPLE_PER_SOURCE, SEED + 40
    )
    actual_test_features = test_data.iloc[test_indices][feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )
    actual_test_ids = template["sample_id"].to_numpy(dtype=np.int32)[test_indices]
    del test_data, template
    gc.collect()

    actual_model, actual_metadata, actual_importance = fit_source_classifier(
        name="actual_train0_49_vs_test",
        source_zero=actual_train_features,
        source_zero_ids=actual_train_ids,
        source_one=actual_test_features,
        source_one_ids=actual_test_ids,
        feature_columns=feature_columns,
        run_dir=run_dir,
    )
    experiment_results["actual_train0_49_vs_test"] = actual_metadata
    importance_frames.append(actual_importance)
    late_probabilities = actual_model.predict_proba(late_features)[:, 1].astype(np.float32)
    pd.DataFrame(
        {
            "sample_id": late_ids,
            "month": late_months,
            "test_probability": late_probabilities,
        }
    ).to_feather(run_dir / "late_train_domain_scores.feather")

    importance_all = pd.concat(importance_frames, ignore_index=True)
    importance_all.to_csv(run_dir / "feature_importance.csv", index=False)
    top_actual = actual_importance.head(20)["feature"].tolist()
    distribution_rows = sampled_distribution(
        "train_0_49", actual_train_features, feature_columns, top_actual
    ) + sampled_distribution("test", actual_test_features, feature_columns, top_actual)
    pd.DataFrame(distribution_rows).to_csv(
        run_dir / "top_feature_distribution.csv", index=False
    )

    top_sets = {
        frame.iloc[0]["experiment"]: set(frame.head(20)["feature"])
        for frame in importance_frames
    }
    overlap = {}
    actual_set = top_sets["actual_train0_49_vs_test"]
    for name, feature_set in top_sets.items():
        if name == "actual_train0_49_vs_test":
            continue
        overlap[name] = {
            "top20_intersection_count": len(actual_set & feature_set),
            "top20_jaccard": len(actual_set & feature_set) / len(actual_set | feature_set),
            "shared_features": sorted(actual_set & feature_set),
        }

    tier_scores = []
    tier_composition = []
    for fold_name in ("train049_valid5059", "train059_valid6070"):
        scores, composition = stratified_oof_report(
            project_dir, fold_name, late_ids, late_months, late_probabilities
        )
        tier_scores.append(scores)
        tier_composition.append(composition)
    pd.concat(tier_scores, ignore_index=True).to_csv(
        run_dir / "oof_scores_by_test_likeness.csv", index=False
    )
    pd.concat(tier_composition, ignore_index=True).to_csv(
        run_dir / "tier_month_composition.csv", index=False
    )

    result = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "diagnostic only; test-likeness is not a replacement validation metric",
        "feature_count": len(feature_columns),
        "excluded_columns": ["sample_id", "month", "target"],
        "source_classifiers": experiment_results,
        "actual_top20_features": top_actual,
        "historical_top20_overlap": overlap,
        "late_domain_probability": {
            "mean": float(late_probabilities.mean()),
            "std": float(late_probabilities.std(ddof=0)),
            "q10": float(np.quantile(late_probabilities, 0.10)),
            "q50": float(np.quantile(late_probabilities, 0.50)),
            "q90": float(np.quantile(late_probabilities, 0.90)),
        },
        "notes": {
            "legacy_static": "older standardized TabM+EXP053R blend, not the current quantile307 checkpoint",
            "gru001": "existing hidden96, 100-step downsampled GRU; diagnostic only",
        },
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
