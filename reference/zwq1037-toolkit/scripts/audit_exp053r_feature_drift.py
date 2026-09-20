"""Low-memory drift audit for the exact 307-feature B001 schema.

The audit reads one feature table at a time, retains a fixed sample from each
labelled month and the test set, and measures distribution / preprocessing
drift without fitting a predictive model.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from train_full_xgboost_exp051_submission import selected_public_features


EXPERIMENT_ID = "EXP-AUDIT-001-EXP053R-DRIFT"
TRAIN_SAMPLES_PER_MONTH = 2000
TEST_SAMPLE_COUNT = 100_000
RANDOM_SEED = 42
QUANTILES = np.asarray([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
WINDOWS = {
    "month_0_39": (0, 39),
    "month_40_49": (40, 49),
    "month_50_59": (50, 59),
    "month_60_70": (60, 70),
    "month_0_70": (0, 70),
}


def schema_columns(path: Path) -> list[str]:
    return list(arrow_dataset.dataset(path, format="feather").schema.names)


def make_samples(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month"],
    )
    generator = np.random.default_rng(RANDOM_SEED)
    selected = []
    month_values = labels["month"].to_numpy()
    for month in range(71):
        indices = np.flatnonzero(month_values == month)
        count = min(TRAIN_SAMPLES_PER_MONTH, len(indices))
        selected.append(generator.choice(indices, size=count, replace=False))
    train_sample = labels.iloc[np.concatenate(selected)].sort_values("sample_id")
    train_sample = train_sample.reset_index(drop=True)

    template = pd.read_csv(
        project_dir / "data" / "raw" / "submission.csv",
        usecols=["sample_id"],
    )
    count = min(TEST_SAMPLE_COUNT, len(template))
    indices = generator.choice(len(template), size=count, replace=False)
    test_sample = template.iloc[indices].sort_values("sample_id").reset_index(drop=True)
    return train_sample, test_sample


def load_sampled_feather(
    path: Path, feature_columns: list[str], sample_rows: pd.DataFrame
) -> pd.DataFrame:
    data = pd.read_feather(path, columns=["sample_id", *feature_columns])
    sampled = sample_rows[["sample_id"]].merge(
        data, on="sample_id", how="left", validate="one_to_one"
    )
    del data
    gc.collect()
    return sampled


def load_sampled_csv(
    path: Path, feature_columns: list[str], sample_rows: pd.DataFrame
) -> pd.DataFrame:
    selected_ids = set(sample_rows["sample_id"].astype(int).tolist())
    pieces = []
    dtypes = {name: np.float32 for name in feature_columns}
    dtypes["sample_id"] = np.int32
    for chunk in pd.read_csv(
        path,
        usecols=["sample_id", *feature_columns],
        dtype=dtypes,
        chunksize=100_000,
    ):
        keep = chunk["sample_id"].isin(selected_ids)
        if bool(keep.any()):
            pieces.append(chunk.loc[keep])
    data = pd.concat(pieces, ignore_index=True)
    sampled = sample_rows[["sample_id"]].merge(
        data, on="sample_id", how="left", validate="one_to_one"
    )
    del data, pieces
    gc.collect()
    return sampled


def finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)].astype(np.float64, copy=False)


def distribution_stats(
    values: np.ndarray,
    training_median: float,
    training_mean: float,
    training_std: float,
) -> dict[str, float | int]:
    finite = finite_values(values)
    quantiles = (
        np.quantile(finite, QUANTILES)
        if len(finite)
        else np.full(len(QUANTILES), np.nan)
    )
    imputed = np.where(np.isfinite(values), values, training_median).astype(
        np.float64, copy=False
    )
    denominator = training_std if training_std >= 1.0e-6 else 1.0
    standardized = (imputed - training_mean) / denominator
    return {
        "rows": int(len(values)),
        "finite_rows": int(len(finite)),
        "missing_rate": float(1.0 - len(finite) / max(len(values), 1)),
        "mean": float(finite.mean()) if len(finite) else np.nan,
        "std": float(finite.std(ddof=0)) if len(finite) else np.nan,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "abs_z_gt_5_rate": float(np.mean(np.abs(standardized) > 5.0)),
        "abs_z_gt_10_rate": float(np.mean(np.abs(standardized) > 10.0)),
    }


def population_stability_index(reference: np.ndarray, comparison: np.ndarray) -> float:
    reference = finite_values(reference)
    comparison = finite_values(comparison)
    if len(reference) == 0 or len(comparison) == 0:
        return np.nan
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, 11)))
    if len(edges) < 3:
        return 0.0 if np.array_equal(np.unique(reference), np.unique(comparison)) else np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    cmp_counts = np.histogram(comparison, bins=edges)[0].astype(np.float64)
    epsilon = 1.0e-6
    ref_share = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cmp_share = np.clip(cmp_counts / cmp_counts.sum(), epsilon, None)
    return float(np.sum((cmp_share - ref_share) * np.log(cmp_share / ref_share)))


def audit_feature_group(
    group_name: str,
    feature_columns: list[str],
    train_values: pd.DataFrame,
    test_values: pd.DataFrame,
    train_sample: pd.DataFrame,
    scaler: dict[str, tuple[float, float, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    statistic_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    months = train_sample["month"].to_numpy()
    for feature in feature_columns:
        all_train = train_values[feature].to_numpy(dtype=np.float64, copy=False)
        test = test_values[feature].to_numpy(dtype=np.float64, copy=False)
        training_median, training_mean, training_std = scaler[feature]
        window_arrays: dict[str, np.ndarray] = {}
        for window_name, (start, end) in WINDOWS.items():
            mask = (months >= start) & (months <= end)
            window_arrays[window_name] = all_train[mask]
        window_arrays["test"] = test

        stats_by_split = {}
        for split_name, values in window_arrays.items():
            stats = distribution_stats(
                values, training_median, training_mean, training_std
            )
            stats_by_split[split_name] = stats
            statistic_rows.append(
                {
                    "feature": feature,
                    "feature_group": group_name,
                    "split": split_name,
                    **stats,
                }
            )

        reference = stats_by_split["month_0_70"]
        test_stats = stats_by_split["test"]
        reference_iqr = reference["q75"] - reference["q25"]
        denominator = max(abs(reference_iqr), training_std * 0.05, 1.0e-12)
        median_shift_iqr = abs(test_stats["q50"] - reference["q50"]) / denominator
        mean_shift_std = abs(test_stats["mean"] - reference["mean"]) / max(
            training_std, 1.0e-12
        )
        test_iqr = test_stats["q75"] - test_stats["q25"]
        iqr_ratio = test_iqr / reference_iqr if abs(reference_iqr) > 1.0e-12 else np.nan
        psi = population_stability_index(window_arrays["month_0_70"], test)
        drift_score = float(
            np.nan_to_num(psi, nan=0.0)
            + min(median_shift_iqr, 10.0) * 0.25
            + min(mean_shift_std, 10.0) * 0.25
            + abs(test_stats["missing_rate"] - reference["missing_rate"]) * 5.0
            + min(test_stats["abs_z_gt_10_rate"], 0.20) * 5.0
        )
        historical_psi = {
            name: population_stability_index(window_arrays["month_0_70"], values)
            for name, values in window_arrays.items()
            if name not in {"month_0_70", "test"}
        }
        drift_rows.append(
            {
                "feature": feature,
                "feature_group": group_name,
                "test_psi_vs_month_0_70": psi,
                "test_median_shift_in_train_iqr": median_shift_iqr,
                "test_mean_shift_in_train_std": mean_shift_std,
                "test_iqr_ratio": iqr_ratio,
                "test_missing_rate": test_stats["missing_rate"],
                "train_missing_rate": reference["missing_rate"],
                "missing_rate_change": test_stats["missing_rate"]
                - reference["missing_rate"],
                "test_abs_z_gt_5_rate": test_stats["abs_z_gt_5_rate"],
                "test_abs_z_gt_10_rate": test_stats["abs_z_gt_10_rate"],
                "month_0_39_psi_vs_full_train": historical_psi["month_0_39"],
                "month_40_49_psi_vs_full_train": historical_psi["month_40_49"],
                "month_50_59_psi_vs_full_train": historical_psi["month_50_59"],
                "month_60_70_psi_vs_full_train": historical_psi["month_60_70"],
                "heuristic_drift_score": drift_score,
            }
        )
    return statistic_rows, drift_rows


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    train_sample, test_sample = make_samples(project_dir)
    train_sample.to_feather(run_dir / "sampled_train_ids.feather")
    test_sample.to_feather(run_dir / "sampled_test_ids.feather")

    public_features, dropped_public_features = selected_public_features(project_dir)
    base_sources = [
        ("market_full", "market_features", None),
        ("market_last60", "market_last60_features_complete", None),
        ("market_level2", "market_level2_features", None),
        ("market_microstructure", "market_microstructure_features", list(BOOK_FEATURE_COLUMNS)),
    ]
    sources: list[tuple[str, str, list[str], str]] = []
    base_features: list[str] = []
    for group_name, file_stem, declared_columns in base_sources:
        train_path = processed_dir / f"train_{file_stem}.feather"
        columns = declared_columns or schema_columns(train_path)[1:]
        sources.append((group_name, file_stem, list(columns), "feather"))
        base_features.extend(columns)
    sources.extend(
        [
            ("transaction_flow", "transaction_flow_features", list(TRANSACTION_FEATURE_COLUMNS), "feather"),
            ("order_multiwindow", "order_multiwindow_features", list(ORDER_MULTI_FEATURE_COLUMNS), "feather"),
            ("transaction_gap", "transaction_event_gap_features", list(GAP_FEATURE_COLUMNS), "feather"),
            ("public_notebook", "public", list(public_features), "csv"),
            ("market_last15", "market_last15_baseline_features", list(LAST15_FEATURE_COLUMNS), "feather"),
        ]
    )
    feature_columns = (
        base_features
        + list(TRANSACTION_FEATURE_COLUMNS)
        + list(ORDER_MULTI_FEATURE_COLUMNS)
        + list(GAP_FEATURE_COLUMNS)
        + list(public_features)
        + list(LAST15_FEATURE_COLUMNS)
    )
    if len(feature_columns) != 307 or len(set(feature_columns)) != 307:
        raise AssertionError("Audit schema must match the 307 unique B001 features.")

    preprocessing_path = (
        project_dir
        / "outputs"
        / "models"
        / "tabm001_tree053r_blend75_fulltrain_tabm_preprocessing.npz"
    )
    preprocessing = np.load(preprocessing_path)
    preprocessing_columns = preprocessing["feature_columns"].astype(str).tolist()
    if preprocessing_columns != feature_columns:
        raise AssertionError("Audit and deployed TabM preprocessing schemas differ.")
    scaler = {
        name: (
            float(preprocessing["medians"][index]),
            float(preprocessing["means"][index]),
            float(preprocessing["standard_deviations"][index]),
        )
        for index, name in enumerate(feature_columns)
    }

    all_statistics: list[dict[str, object]] = []
    all_drift: list[dict[str, object]] = []
    for group_name, file_stem, columns, source_type in sources:
        print(f"auditing {group_name}: {len(columns)} features", flush=True)
        if source_type == "csv":
            root = project_dir / "data" / "interim" / "public_features" / "rfmf_0726data"
            train_values = load_sampled_csv(root / "train.csv", columns, train_sample)
            test_values = load_sampled_csv(root / "test.csv", columns, test_sample)
        else:
            train_values = load_sampled_feather(
                processed_dir / f"train_{file_stem}.feather", columns, train_sample
            )
            test_values = load_sampled_feather(
                processed_dir / f"test_{file_stem}.feather", columns, test_sample
            )
        statistic_rows, drift_rows = audit_feature_group(
            group_name,
            columns,
            train_values,
            test_values,
            train_sample,
            scaler,
        )
        all_statistics.extend(statistic_rows)
        all_drift.extend(drift_rows)
        del train_values, test_values
        gc.collect()

    statistics = pd.DataFrame(all_statistics)
    drift = pd.DataFrame(all_drift).sort_values(
        "heuristic_drift_score", ascending=False
    )
    statistics.to_csv(run_dir / "feature_distribution_by_split.csv", index=False)
    drift.to_csv(run_dir / "feature_drift_ranking.csv", index=False)

    group_summary = (
        drift.groupby("feature_group", as_index=False)
        .agg(
            feature_count=("feature", "size"),
            median_test_psi=("test_psi_vs_month_0_70", "median"),
            q90_test_psi=("test_psi_vs_month_0_70", lambda values: values.quantile(0.90)),
            mean_clip10_rate=("test_abs_z_gt_10_rate", "mean"),
            max_clip10_rate=("test_abs_z_gt_10_rate", "max"),
            mean_abs_missing_change=("missing_rate_change", lambda values: values.abs().mean()),
        )
        .sort_values("q90_test_psi", ascending=False)
    )
    group_summary.to_csv(run_dir / "feature_group_summary.csv", index=False)

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "unsupervised train/validation/test feature drift and TabM clipping audit",
        "feature_count": len(feature_columns),
        "train_samples_per_month": TRAIN_SAMPLES_PER_MONTH,
        "sampled_train_rows": int(len(train_sample)),
        "sampled_test_rows": int(len(test_sample)),
        "random_seed": RANDOM_SEED,
        "preprocessing_reference": str(preprocessing_path.relative_to(project_dir)),
        "preprocessing_fit_months": "0-70 fulltrain B001",
        "public_feature_selection_note": (
            "uses the deployed EXP051 public feature list; this audit diagnoses "
            "distribution drift but does not claim a new blind feature selection"
        ),
        "dropped_public_features": dropped_public_features,
        "drift_score_note": (
            "heuristic ranking for diagnosis only; high drift does not imply a "
            "feature should be removed"
        ),
        "top_30_features": drift.head(30).to_dict(orient="records"),
        "group_summary": group_summary.to_dict(orient="records"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(group_summary.to_string(index=False), flush=True)
    print("\nTop 30 drift features:", flush=True)
    print(
        drift.head(30)[
            [
                "feature",
                "feature_group",
                "test_psi_vs_month_0_70",
                "test_median_shift_in_train_iqr",
                "missing_rate_change",
                "test_abs_z_gt_10_rate",
                "heuristic_drift_score",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
