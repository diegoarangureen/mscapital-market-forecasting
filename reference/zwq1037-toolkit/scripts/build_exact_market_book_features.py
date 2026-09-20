"""Rebuild the 35 selected book features from the complete raw market sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from build_market_aggregate_features import safe_write
from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS


def ofi_expressions(level: int) -> list[pl.Expr]:
    bid_price = pl.col(f"bid_price_{level}")
    ask_price = pl.col(f"ask_price_{level}")
    bid_volume = pl.col(f"bid_volume_{level}").cast(pl.Float64)
    ask_volume = pl.col(f"ask_volume_{level}").cast(pl.Float64)
    previous_bid_price = bid_price.shift(1).over("sample_id")
    previous_ask_price = ask_price.shift(1).over("sample_id")
    previous_bid_volume = bid_volume.shift(1).over("sample_id")
    previous_ask_volume = ask_volume.shift(1).over("sample_id")
    previous_mid = (previous_bid_price + previous_ask_price) * 0.5
    current_mid = (bid_price + ask_price) * 0.5
    previous_quote_valid = (
        (previous_mid >= 0.1)
        & (previous_ask_price >= previous_bid_price)
        & ((previous_ask_price - previous_bid_price) / previous_mid <= 0.1)
    )
    current_quote_valid = (
        (current_mid >= 0.1)
        & (ask_price >= bid_price)
        & ((ask_price - bid_price) / current_mid <= 0.1)
    )
    depth = previous_bid_volume + bid_volume + previous_ask_volume + ask_volume
    valid = (
        previous_quote_valid
        & current_quote_valid
        & (previous_bid_volume > 0)
        & (bid_volume > 0)
        & (previous_ask_volume > 0)
        & (ask_volume > 0)
        & (depth > 0)
    )
    bid_flow = (
        pl.when(bid_price >= previous_bid_price).then(bid_volume).otherwise(0.0)
        - pl.when(bid_price <= previous_bid_price)
        .then(previous_bid_volume)
        .otherwise(0.0)
    )
    ask_flow = (
        pl.when(ask_price <= previous_ask_price).then(ask_volume).otherwise(0.0)
        - pl.when(ask_price >= previous_ask_price)
        .then(previous_ask_volume)
        .otherwise(0.0)
    )
    return [
        pl.when(valid)
        .then((bid_flow - ask_flow) / depth)
        .otherwise(None)
        .alias(f"_ofi_{level}")
    ]


def build_features(project_dir: Path, split: str) -> pl.DataFrame:
    market_path = project_dir / "data" / "raw" / split / "market.feather"
    data = (
        pl.scan_ipc(market_path)
        .select(
            "sample_id",
            "seconds_before_predict",
            "ask_price_1",
            "ask_volume_1",
            "bid_price_1",
            "bid_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "bid_price_2",
            "bid_volume_2",
        )
        .sort(
            ["sample_id", "seconds_before_predict"],
            descending=[False, True],
        )
        .with_columns(
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
            (pl.col("ask_volume_1") + pl.col("bid_volume_1"))
            .cast(pl.Float64)
            .alias("_depth_1"),
            (pl.col("ask_volume_2") + pl.col("bid_volume_2"))
            .cast(pl.Float64)
            .alias("_depth_2"),
        )
        .with_columns(
            (
                (pl.col("_mid") >= 0.1)
                & (pl.col("ask_price_1") >= pl.col("bid_price_1"))
                & (
                    (pl.col("ask_price_1") - pl.col("bid_price_1"))
                    / pl.col("_mid")
                    <= 0.1
                )
            ).alias("_valid_mid"),
            (
                (pl.col("ask_volume_1") > 0) & (pl.col("bid_volume_1") > 0)
            ).alias("_valid_depth_1"),
            (
                (pl.col("ask_volume_2") > 0) & (pl.col("bid_volume_2") > 0)
            ).alias("_valid_depth_2"),
        )
        .with_columns(
            pl.when(pl.col("_valid_depth_1"))
            .then(
                (pl.col("bid_volume_1") - pl.col("ask_volume_1"))
                / pl.col("_depth_1")
            )
            .otherwise(None)
            .alias("_imbalance_1"),
            pl.when(pl.col("_valid_depth_2"))
            .then(
                (pl.col("bid_volume_2") - pl.col("ask_volume_2"))
                / pl.col("_depth_2")
            )
            .otherwise(None)
            .alias("_imbalance_2"),
            pl.when(pl.col("_valid_depth_1") & pl.col("_valid_depth_2"))
            .then(
                (
                    pl.col("bid_volume_1")
                    + pl.col("bid_volume_2")
                    - pl.col("ask_volume_1")
                    - pl.col("ask_volume_2")
                )
                / (pl.col("_depth_1") + pl.col("_depth_2"))
            )
            .otherwise(None)
            .alias("_total_imbalance"),
            pl.when(pl.col("_valid_depth_1") & pl.col("_valid_mid"))
            .then(
                (
                    (
                        pl.col("ask_price_1") * pl.col("bid_volume_1")
                        + pl.col("bid_price_1") * pl.col("ask_volume_1")
                    )
                    / pl.col("_depth_1")
                    - pl.col("_mid")
                )
                / pl.col("_mid")
            )
            .otherwise(None)
            .alias("_microprice_displacement"),
            pl.when(pl.col("_valid_depth_1") & pl.col("_valid_depth_2"))
            .then((pl.col("_depth_1") + pl.col("_depth_2")).log1p())
            .otherwise(None)
            .alias("_log_total_depth"),
            *ofi_expressions(1),
            *ofi_expressions(2),
        )
        .with_columns(
            pl.when(pl.col("_ofi_1").is_not_null() & pl.col("_ofi_2").is_not_null())
            .then(pl.col("_ofi_1") + pl.col("_ofi_2"))
            .otherwise(None)
            .alias("_multilevel_ofi")
        )
    )
    seconds = pl.col("seconds_before_predict")
    in_60 = seconds <= 60.0
    in_20 = seconds <= 20.0
    in_previous_40 = (seconds > 20.0) & (seconds <= 60.0)

    def latest(column: str) -> pl.Expr:
        return pl.col(column).sort_by(seconds).drop_nulls().first()

    aggregates = data.group_by("sample_id").agg(
        pl.col("_imbalance_1").mean().alias("book_imbalance_1_mean_robust"),
        pl.col("_imbalance_1").std(ddof=0).alias("book_imbalance_1_std_robust"),
        latest("_imbalance_1").alias("book_imbalance_1_last_robust"),
        pl.col("_imbalance_1")
        .filter(in_60)
        .mean()
        .alias("book_imbalance_1_60_mean_robust"),
        pl.col("_imbalance_1")
        .filter(in_20)
        .mean()
        .alias("book_imbalance_1_20_mean_robust"),
        pl.col("_imbalance_2").mean().alias("book_imbalance_2_mean_robust"),
        pl.col("_imbalance_2").std(ddof=0).alias("book_imbalance_2_std_robust"),
        latest("_imbalance_2").alias("book_imbalance_2_last_robust"),
        pl.col("_imbalance_2")
        .filter(in_60)
        .mean()
        .alias("book_imbalance_2_60_mean_robust"),
        pl.col("_total_imbalance").mean().alias("total_book_imbalance_mean"),
        latest("_total_imbalance").alias("total_book_imbalance_last"),
        pl.col("_total_imbalance")
        .filter(in_60)
        .mean()
        .alias("total_book_imbalance_60_mean"),
        pl.col("_microprice_displacement").mean().alias("microprice_displacement_mean"),
        pl.col("_microprice_displacement")
        .std(ddof=0)
        .alias("microprice_displacement_std"),
        latest("_microprice_displacement").alias("microprice_displacement_last"),
        pl.col("_microprice_displacement")
        .filter(in_60)
        .mean()
        .alias("microprice_displacement_60_mean"),
        pl.col("_microprice_displacement")
        .filter(in_20)
        .mean()
        .alias("microprice_displacement_20_mean"),
        pl.col("_ofi_1").mean().alias("ofi_1_mean"),
        pl.col("_ofi_1").std(ddof=0).alias("ofi_1_std"),
        pl.col("_ofi_1").sum().alias("ofi_1_sum"),
        latest("_ofi_1").alias("ofi_1_last"),
        pl.col("_ofi_1").filter(in_60).mean().alias("ofi_1_60_mean"),
        pl.col("_ofi_1").filter(in_20).mean().alias("ofi_1_20_mean"),
        (
            pl.col("_ofi_1").filter(in_20).mean()
            - pl.col("_ofi_1").filter(in_previous_40).mean()
        ).alias("ofi_1_acceleration"),
        pl.col("_ofi_2").mean().alias("ofi_2_mean"),
        pl.col("_ofi_2").sum().alias("ofi_2_sum"),
        pl.col("_ofi_2").filter(in_60).mean().alias("ofi_2_60_mean"),
        pl.col("_multilevel_ofi").mean().alias("multilevel_ofi_mean"),
        pl.col("_multilevel_ofi").sum().alias("multilevel_ofi_sum"),
        pl.col("_multilevel_ofi")
        .filter(in_60)
        .mean()
        .alias("multilevel_ofi_60_mean"),
        (
            pl.col("_multilevel_ofi").filter(in_20).mean()
            - pl.col("_multilevel_ofi").filter(in_previous_40).mean()
        ).alias("multilevel_ofi_acceleration"),
        pl.col("_log_total_depth").mean().alias("log_total_depth_mean"),
        pl.col("_log_total_depth").std(ddof=0).alias("log_total_depth_std"),
        latest("_log_total_depth").alias("log_total_depth_last"),
        pl.col("_log_total_depth")
        .filter(in_60)
        .mean()
        .alias("log_total_depth_60_mean"),
    )
    output = aggregates.select("sample_id", *BOOK_FEATURE_COLUMNS).collect(
        engine="streaming"
    )
    if output.columns[1:] != BOOK_FEATURE_COLUMNS:
        raise AssertionError("Exact book feature schema differs from selected schema.")
    values = output.select(BOOK_FEATURE_COLUMNS).to_numpy()
    if np.isinf(values).any():
        raise AssertionError("Exact book features contain infinity.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    output = build_features(project_dir, arguments.split)
    output_path = (
        project_dir
        / "data"
        / "processed"
        / f"{arguments.split}_market_book_features_exact.feather"
    )
    safe_write(output, output_path)
    print(f"shape = {output.shape}", flush=True)
    print(f"missing values = {int(output.select(BOOK_FEATURE_COLUMNS).null_count().sum_horizontal().item())}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
