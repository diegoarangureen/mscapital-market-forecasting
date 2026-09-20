"""Build three exact within-sample liquidity tail-shape features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from build_market_aggregate_features import safe_write


FEATURE_COLUMNS = [
    "spread_tail_ratio_full",
    "depth_drought_ratio_full",
    "trade_volume_concentration_60",
]


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator.abs() > 1e-12).then(
        numerator / denominator
    ).otherwise(None)


def build_features(project_dir: Path, split: str) -> pl.DataFrame:
    raw_dir = project_dir / "data" / "raw" / split
    market = (
        pl.scan_ipc(raw_dir / "market.feather")
        .select(
            "sample_id",
            "ask_price_1",
            "bid_price_1",
            "ask_volume_1",
            "bid_volume_1",
        )
        .with_columns(
            (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("_spread"),
            (pl.col("ask_volume_1") + pl.col("bid_volume_1"))
            .cast(pl.Float64)
            .alias("_depth"),
        )
        .group_by("sample_id")
        .agg(
            pl.col("_spread").quantile(0.90, interpolation="linear").alias("_sp_q90"),
            pl.col("_spread").quantile(0.50, interpolation="linear").alias("_sp_q50"),
            pl.col("_depth").quantile(0.10, interpolation="linear").alias("_depth_q10"),
            pl.col("_depth").quantile(0.50, interpolation="linear").alias("_depth_q50"),
        )
        .with_columns(
            safe_ratio(pl.col("_sp_q90"), pl.col("_sp_q50")).alias(
                "spread_tail_ratio_full"
            ),
            safe_ratio(pl.col("_depth_q10"), pl.col("_depth_q50")).alias(
                "depth_drought_ratio_full"
            ),
        )
        .select("sample_id", "spread_tail_ratio_full", "depth_drought_ratio_full")
        .collect(engine="streaming")
    )
    transaction = (
        pl.scan_ipc(raw_dir / "transaction.feather")
        .select("sample_id", "volume")
        .group_by("sample_id")
        .agg(
            pl.col("volume").max().cast(pl.Float64).alias("_max_volume"),
            pl.col("volume").sum().cast(pl.Float64).alias("_total_volume"),
        )
        .with_columns(
            safe_ratio(pl.col("_max_volume"), pl.col("_total_volume") + 1.0).alias(
                "trade_volume_concentration_60"
            )
        )
        .select("sample_id", "trade_volume_concentration_60")
        .collect(engine="streaming")
    )
    sample_ids = pl.read_ipc(
        project_dir / "data" / "processed" / f"{split}_market_features.feather",
        columns=["sample_id"],
    )
    output = (
        sample_ids.join(market, on="sample_id", how="left")
        .join(transaction, on="sample_id", how="left")
        .with_columns(pl.col("trade_volume_concentration_60").fill_null(0.0))
        .select("sample_id", *FEATURE_COLUMNS)
    )
    values = output.select(FEATURE_COLUMNS).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("Liquidity tail features contain missing or infinite values.")
    return output


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
        / f"{arguments.split}_liquidity_tail_shape_features.feather"
    )
    safe_write(data, output_path)
    print(f"shape = {data.shape}", flush=True)
    print(f"summary = {data.select(FEATURE_COLUMNS).describe()}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
