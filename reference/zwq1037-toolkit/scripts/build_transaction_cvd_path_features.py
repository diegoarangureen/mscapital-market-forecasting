"""Build two non-redundant CVD path-shape features per sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from build_market_last15_baseline_features import safe_write


FEATURE_COLUMNS = [
    "cvd_range_norm_60",
    "cvd_pullback_from_high_60",
]


def build_features(project_dir: Path, split: str) -> pd.DataFrame:
    transaction_path = project_dir / "data" / "raw" / split / "transaction.feather"
    sample_id_path = (
        project_dir / "data" / "processed" / f"{split}_market_features.feather"
    )
    events = (
        pl.scan_ipc(transaction_path)
        .select("sample_id", "seconds_before_predict", "volume", "side")
        .sort(
            ["sample_id", "seconds_before_predict"],
            descending=[False, True],
        )
        .with_columns(
            pl.when(pl.col("side") == 0)
            .then(pl.col("volume").cast(pl.Float64))
            .otherwise(-pl.col("volume").cast(pl.Float64))
            .alias("_signed_volume")
        )
        .with_columns(
            pl.col("_signed_volume")
            .cum_sum()
            .over("sample_id")
            .alias("_cvd")
        )
    )
    aggregates = (
        events.group_by("sample_id")
        .agg(
            pl.col("volume").sum().cast(pl.Float64).alias("_total_volume"),
            pl.col("_cvd").max().alias("_cvd_max_raw"),
            pl.col("_cvd").min().alias("_cvd_min_raw"),
            pl.col("_cvd")
            .sort_by("seconds_before_predict", descending=True)
            .last()
            .alias("_cvd_final"),
        )
        .with_columns(
            pl.max_horizontal("_cvd_max_raw", pl.lit(0.0)).alias("_cvd_max"),
            pl.min_horizontal("_cvd_min_raw", pl.lit(0.0)).alias("_cvd_min"),
        )
        .with_columns(
            (
                (pl.col("_cvd_max") - pl.col("_cvd_min"))
                / (pl.col("_total_volume") + 1.0)
            ).alias("cvd_range_norm_60"),
            (
                (pl.col("_cvd_max") - pl.col("_cvd_final"))
                / (pl.col("_total_volume") + 1.0)
            ).alias("cvd_pullback_from_high_60"),
        )
        .select("sample_id", *FEATURE_COLUMNS)
        .collect(engine="streaming")
    )
    sample_ids = pl.read_ipc(sample_id_path, columns=["sample_id"])
    complete = sample_ids.join(aggregates, on="sample_id", how="left").with_columns(
        pl.col(FEATURE_COLUMNS).fill_null(0.0)
    )
    data = complete.to_pandas()
    for column in FEATURE_COLUMNS:
        data[column] = data[column].astype(np.float32)
    if data["sample_id"].duplicated().any() or data[FEATURE_COLUMNS].isna().any().any():
        raise AssertionError("CVD feature table has duplicate IDs or missing values.")
    if np.isinf(data[FEATURE_COLUMNS].to_numpy(copy=False)).any():
        raise AssertionError("CVD feature table contains infinity.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    data = build_features(project_dir, arguments.split)
    output_path = (
        project_dir
        / "data"
        / "processed"
        / f"{arguments.split}_transaction_cvd_path_features.feather"
    )
    safe_write(data, output_path)
    print(f"shape = {data.shape}", flush=True)
    print(f"zero rates = {(data[FEATURE_COLUMNS] == 0).mean().to_dict()}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
