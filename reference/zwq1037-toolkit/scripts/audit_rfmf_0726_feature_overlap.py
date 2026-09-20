"""Audit public RFMF-0726 features against EXP037 without using validation targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS


SAMPLE_MODULUS = 10
PUBLIC_CSV_RELATIVE = Path(
    "data/interim/public_features/rfmf_0726data/train.csv"
)
MANUAL_EQUIVALENTS = {
    "t_avg_time_gap": "trade_event_gap_mean",
    "t_time_gap_std": "trade_event_gap_std",
    "t_max_time_gap": "trade_event_gap_max",
}


def sampled_ipc(path: Path) -> pd.DataFrame:
    """Read a deterministic ten-percent sample from one IPC feature table."""

    return (
        pl.scan_ipc(path)
        .filter((pl.col("sample_id") % SAMPLE_MODULUS) == 0)
        .collect()
        .to_pandas()
    )


def load_exp037_sample(project_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the exact EXP037 feature groups for the deterministic sample."""

    processed = project_dir / "data" / "processed"
    table_paths = [
        processed / "train_market_features.feather",
        processed / "train_market_last60_features_complete.feather",
        processed / "train_market_level2_features.feather",
        processed / "train_market_microstructure_features.feather",
        processed / "train_transaction_flow_features.feather",
        processed / "train_order_multiwindow_features.feather",
        processed / "train_transaction_event_gap_features.feather",
    ]
    tables = [sampled_ipc(path) for path in table_paths]
    data = tables[0]
    for table in tables[1:]:
        data = data.merge(table, on="sample_id", how="left", validate="one_to_one")

    # 读取列名本身不接触目标，确保审计使用的正是 EXP037 的 159 列。
    # Read schemas only to reconstruct the exact 159-column EXP037 feature list.
    base_features = (
        list(pl.read_ipc_schema(table_paths[0]))[1:]
        + list(pl.read_ipc_schema(table_paths[1]))[1:]
        + list(pl.read_ipc_schema(table_paths[2]))[1:]
        + BOOK_FEATURE_COLUMNS
    )
    feature_columns = (
        base_features
        + TRANSACTION_FEATURE_COLUMNS
        + ORDER_MULTI_FEATURE_COLUMNS
        + GAP_FEATURE_COLUMNS
    )
    if len(feature_columns) != 159 or len(set(feature_columns)) != 159:
        raise AssertionError("Unexpected EXP037 audit schema.")
    return data[["sample_id", *feature_columns]], feature_columns


def load_public_sample(project_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Stream the large public CSV and retain the same deterministic sample."""

    csv_path = project_dir / PUBLIC_CSV_RELATIVE
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    feature_columns = [name for name in header if name not in {"sample_id", "target"}]
    dtypes = {name: np.float32 for name in feature_columns}
    dtypes["sample_id"] = np.int32
    chunks = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["sample_id", *feature_columns],
        dtype=dtypes,
        chunksize=100_000,
    ):
        chunks.append(chunk.loc[(chunk["sample_id"] % SAMPLE_MODULUS) == 0])
    sample = pd.concat(chunks, ignore_index=True)
    if sample["sample_id"].nunique() != len(sample):
        raise AssertionError("Public sample IDs are not unique.")
    return sample, feature_columns


def standardized_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute and standardize columns for a target-free correlation audit."""

    values = frame.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    means = np.nanmean(values, axis=0)
    standard_deviations = np.nanstd(values, axis=0)
    constants = ~np.isfinite(standard_deviations) | (standard_deviations < 1e-12)
    standard_deviations[constants] = 1.0
    values = (values - means) / standard_deviations
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32), constants


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    labels = sampled_ipc(project_dir / "data" / "raw" / "label.feather")
    labels = labels.loc[labels["month"] <= 59, ["sample_id", "month"]]
    ours, our_features = load_exp037_sample(project_dir)
    public, public_features = load_public_sample(project_dir)
    audit_data = (
        labels.merge(ours, on="sample_id", how="inner", validate="one_to_one")
        .merge(public, on="sample_id", how="inner", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    if len(audit_data) < 90_000:
        raise AssertionError(f"Too few training-only audit rows: {len(audit_data)}.")

    our_matrix, our_constants = standardized_matrix(audit_data[our_features])
    public_matrix, public_constants = standardized_matrix(audit_data[public_features])
    correlations = (our_matrix.T @ public_matrix) / float(len(audit_data))
    absolute_correlations = np.abs(correlations)
    closest_indices = absolute_correlations.argmax(axis=0)
    closest_values = correlations[closest_indices, np.arange(len(public_features))]

    rows = []
    for index, public_feature in enumerate(public_features):
        closest_feature = our_features[int(closest_indices[index])]
        maximum = float(closest_values[index])
        manual_match = MANUAL_EQUIVALENTS.get(public_feature)
        drop_reason = ""
        if public_constants[index]:
            drop_reason = "constant_on_training_sample"
        elif manual_match is not None:
            drop_reason = f"same_formula_family_as:{manual_match}"
        elif abs(maximum) >= 0.995:
            drop_reason = f"abs_correlation_ge_0.995:{closest_feature}"
        rows.append(
            {
                "public_feature": public_feature,
                "closest_exp037_feature": closest_feature,
                "correlation": maximum,
                "absolute_correlation": abs(maximum),
                "manual_equivalent": manual_match or "",
                "drop_reason": drop_reason,
                "keep_for_incremental_test": drop_reason == "",
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        "absolute_correlation", ascending=False
    )
    output_dir = (
        project_dir / "data" / "interim" / "tree_experiments" / "EXP-TREE-050"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "public_feature_overlap_audit.csv"
    audit.to_csv(audit_path, index=False)
    kept = audit.loc[audit["keep_for_incremental_test"], "public_feature"].tolist()
    summary = {
        "audit_scope": "month 0-59 only; deterministic sample_id modulo 10 sample",
        "audit_rows": len(audit_data),
        "exp037_feature_count": len(our_features),
        "public_feature_count": len(public_features),
        "public_constant_count": int(public_constants.sum()),
        "public_abs_correlation_ge_0_995_count": int(
            (audit["absolute_correlation"] >= 0.995).sum()
        ),
        "manual_equivalent_count": len(MANUAL_EQUIVALENTS),
        "kept_public_feature_count": len(kept),
        "kept_public_features": kept,
    }
    summary_path = output_dir / "overlap_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(audit.head(25).to_string(index=False), flush=True)
    print(f"audit={audit_path}", flush=True)


if __name__ == "__main__":
    main()
