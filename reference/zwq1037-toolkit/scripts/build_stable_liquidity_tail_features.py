"""Replace the unstable spread ratio with a mid-normalized spread-tail excess."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from build_market_aggregate_features import safe_write


FEATURE_COLUMNS = [
    "spread_tail_excess_relative_full",
    "depth_drought_ratio_full",
    "trade_volume_concentration_60",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    raw_dir = project_dir / "data" / "raw" / arguments.split
    processed_dir = project_dir / "data" / "processed"

    stable_spread = (
        pl.scan_ipc(raw_dir / "market.feather")
        .select("sample_id", "ask_price_1", "bid_price_1")
        .with_columns(
            (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("_spread"),
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
        )
        .group_by("sample_id")
        .agg(
            pl.col("_spread").quantile(0.90, interpolation="linear").alias("_sp_q90"),
            pl.col("_spread").quantile(0.50, interpolation="linear").alias("_sp_q50"),
            pl.col("_mid").quantile(0.50, interpolation="linear").alias("_mid_q50"),
        )
        .with_columns(
            (
                (pl.col("_sp_q90") - pl.col("_sp_q50"))
                / (pl.col("_mid_q50").abs() + 1e-8)
            ).alias("spread_tail_excess_relative_full")
        )
        .select("sample_id", "spread_tail_excess_relative_full")
        .collect(engine="streaming")
    )
    earlier = pl.read_ipc(
        processed_dir / f"{arguments.split}_liquidity_tail_shape_features.feather",
        columns=[
            "sample_id",
            "depth_drought_ratio_full",
            "trade_volume_concentration_60",
        ],
    )
    output = earlier.join(stable_spread, on="sample_id", how="left").select(
        "sample_id", *FEATURE_COLUMNS
    )
    values = output.select(FEATURE_COLUMNS).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("Stable liquidity tail features are not finite.")
    output_path = (
        processed_dir
        / f"{arguments.split}_stable_liquidity_tail_shape_features.feather"
    )
    safe_write(output, output_path)
    print(f"shape = {output.shape}", flush=True)
    print(f"summary = {output.select(FEATURE_COLUMNS).describe()}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
