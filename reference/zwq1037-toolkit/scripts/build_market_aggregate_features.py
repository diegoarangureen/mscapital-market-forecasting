"""Build V1, last-60, and L2 aggregate market features in one full scan."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import polars as pl


def safe_write(data: pl.DataFrame, output_path: Path) -> None:
    """Write Feather through an ASCII temporary path on Windows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        data.write_ipc(temporary_path)
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def filtered(column_name: str, recent: pl.Expr) -> pl.Expr:
    """Select one source column only inside the final 60 seconds."""

    return pl.col(column_name).filter(recent)


def build_aggregate_table(market_path: Path) -> pl.DataFrame:
    """Aggregate all required market columns by sample in one lazy scan."""

    seconds = pl.col("seconds_before_predict")
    recent = seconds <= 60.0
    expressions = [
        pl.len().alias("market_row_count"),
        seconds.max().alias("seconds_max"),
        seconds.min().alias("seconds_min"),
        (seconds.diff().abs() > 4.5).sum().alias("long_gap_count"),
        pl.col("transaction_count").sum().alias("transaction_count_sum"),
        pl.col("transaction_count").mean().alias("transaction_count_mean"),
        pl.col("transaction_count").max().alias("transaction_count_max"),
        (pl.col("transaction_count") > 0).mean().alias("has_transaction_ratio"),
        pl.col("transaction_volume").sum().alias("transaction_volume_sum"),
        pl.col("transaction_volume").mean().alias("transaction_volume_mean"),
        pl.col("transaction_volume").max().alias("transaction_volume_max"),
        pl.col("transaction_avgprice").mean().alias("transaction_avgprice_mean"),
        pl.col("transaction_avgprice").std().alias("transaction_avgprice_std"),
        pl.col("transaction_avgprice").min().alias("transaction_avgprice_min"),
        pl.col("transaction_avgprice").max().alias("transaction_avgprice_max"),
        pl.col("transaction_avgprice").drop_nulls().first().alias("transaction_avgprice_first_valid"),
        pl.col("transaction_avgprice").drop_nulls().last().alias("transaction_avgprice_last_valid"),
        recent.sum().alias("last60_row_count"),
        filtered("transaction_count", recent).sum().alias("last60_transaction_count_sum"),
        filtered("transaction_count", recent).max().alias("last60_transaction_count_max"),
        (filtered("transaction_count", recent) > 0).mean().alias("last60_has_transaction_ratio"),
        filtered("transaction_volume", recent).sum().alias("last60_transaction_volume_sum"),
        filtered("transaction_volume", recent).max().alias("last60_transaction_volume_max"),
        filtered("transaction_avgprice", recent).mean().alias("last60_transaction_avgprice_mean"),
        filtered("transaction_avgprice", recent).std().alias("last60_transaction_avgprice_std"),
        filtered("transaction_avgprice", recent).drop_nulls().last().alias("last60_transaction_avgprice_last_valid"),
    ]
    for level in [1, 2]:
        for side in ["ask", "bid"]:
            price_name = f"{side}_price_{level}"
            volume_name = f"{side}_volume_{level}"
            price = pl.col(price_name)
            volume = pl.col(volume_name)
            expressions.extend(
                [
                    price.mean().alias(f"{price_name}_mean"),
                    price.std().alias(f"{price_name}_std"),
                    price.min().alias(f"{price_name}_min"),
                    price.max().alias(f"{price_name}_max"),
                    price.first().alias(f"{price_name}_first"),
                    price.last().alias(f"{price_name}_last"),
                    volume.sum().alias(f"{volume_name}_sum"),
                    volume.mean().alias(f"{volume_name}_mean"),
                    volume.max().alias(f"{volume_name}_max"),
                    filtered(price_name, recent).mean().alias(f"last60_{price_name}_mean"),
                    filtered(price_name, recent).std().alias(f"last60_{price_name}_std"),
                    filtered(price_name, recent).first().alias(f"last60_{price_name}_start"),
                    filtered(price_name, recent).last().alias(f"last60_{price_name}_end"),
                    filtered(volume_name, recent).mean().alias(f"last60_{volume_name}_mean"),
                ]
            )
    return (
        pl.scan_ipc(market_path)
        .group_by("sample_id", maintain_order=True)
        .agg(expressions)
        .collect(engine="streaming")
    )


def derive_and_split(data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create derived columns and return the exact three historical schemas."""

    data = data.with_columns(
        (pl.col("ask_price_1_mean") - pl.col("bid_price_1_mean")).alias("spread_1_mean"),
        ((pl.col("ask_price_1_mean") + pl.col("bid_price_1_mean")) / 2.0).alias("mid_price_1_mean"),
        (pl.col("seconds_max") - pl.col("seconds_min")).alias("seconds_coverage"),
        (
            (
                pl.col("ask_price_1_last")
                + pl.col("bid_price_1_last")
                - pl.col("ask_price_1_first")
                - pl.col("bid_price_1_first")
            )
            / 2.0
        ).alias("mid_price_1_change"),
        (
            (pl.col("bid_volume_1_sum") - pl.col("ask_volume_1_sum"))
            / (pl.col("bid_volume_1_sum") + pl.col("ask_volume_1_sum"))
        ).alias("book_volume_imbalance_1"),
        (pl.col("last60_ask_price_1_mean") - pl.col("last60_bid_price_1_mean")).alias("last60_spread_1_mean"),
        (
            (
                pl.col("last60_ask_price_1_end")
                + pl.col("last60_bid_price_1_end")
                - pl.col("last60_ask_price_1_start")
                - pl.col("last60_bid_price_1_start")
            )
            / 2.0
        ).alias("last60_mid_price_1_change"),
        (
            (pl.col("last60_bid_volume_1_mean") - pl.col("last60_ask_volume_1_mean"))
            / (pl.col("last60_bid_volume_1_mean") + pl.col("last60_ask_volume_1_mean"))
        ).alias("last60_book_volume_imbalance_1"),
    )
    full_volume_denominator = pl.col("bid_volume_2_sum") + pl.col("ask_volume_2_sum")
    recent_volume_denominator = pl.col("last60_bid_volume_2_mean") + pl.col("last60_ask_volume_2_mean")
    data = data.with_columns(
        (pl.col("ask_price_2_mean") - pl.col("bid_price_2_mean")).alias("spread_2_mean"),
        ((pl.col("ask_price_2_mean") + pl.col("bid_price_2_mean")) / 2.0).alias("mid_price_2_mean"),
        (
            (
                pl.col("ask_price_2_last")
                + pl.col("bid_price_2_last")
                - pl.col("ask_price_2_first")
                - pl.col("bid_price_2_first")
            )
            / 2.0
        ).alias("mid_price_2_change"),
        pl.when(full_volume_denominator != 0)
        .then((pl.col("bid_volume_2_sum") - pl.col("ask_volume_2_sum")) / full_volume_denominator)
        .otherwise(None)
        .alias("book_volume_imbalance_2"),
        (pl.col("last60_ask_price_2_mean") - pl.col("last60_bid_price_2_mean")).alias("last60_spread_2_mean"),
        (
            (
                pl.col("last60_ask_price_2_end")
                + pl.col("last60_bid_price_2_end")
                - pl.col("last60_ask_price_2_start")
                - pl.col("last60_bid_price_2_start")
            )
            / 2.0
        ).alias("last60_mid_price_2_change"),
        pl.when(recent_volume_denominator != 0)
        .then(
            (pl.col("last60_bid_volume_2_mean") - pl.col("last60_ask_volume_2_mean"))
            / recent_volume_denominator
        )
        .otherwise(None)
        .alias("last60_book_volume_imbalance_2"),
    )
    v1_columns = [
        "sample_id", "market_row_count", "seconds_max", "seconds_min", "long_gap_count",
        "transaction_count_sum", "transaction_count_mean", "transaction_count_max", "has_transaction_ratio",
        "transaction_volume_sum", "transaction_volume_mean", "transaction_volume_max",
        "transaction_avgprice_mean", "transaction_avgprice_std", "transaction_avgprice_min", "transaction_avgprice_max",
        "transaction_avgprice_first_valid", "transaction_avgprice_last_valid",
    ]
    for side in ["ask", "bid"]:
        for kind in ["price", "volume"]:
            name = f"{side}_{kind}_1"
            suffixes = ["mean", "std", "min", "max", "first", "last"] if kind == "price" else ["sum", "mean", "max"]
            v1_columns.extend(f"{name}_{suffix}" for suffix in suffixes)
    v1_columns.extend(["spread_1_mean", "mid_price_1_mean", "seconds_coverage", "mid_price_1_change", "book_volume_imbalance_1"])
    last60_columns = [
        "sample_id", "last60_row_count", "last60_transaction_count_sum", "last60_transaction_count_max",
        "last60_has_transaction_ratio", "last60_transaction_volume_sum", "last60_transaction_volume_max",
        "last60_transaction_avgprice_mean", "last60_transaction_avgprice_std", "last60_transaction_avgprice_last_valid",
    ]
    for side in ["ask", "bid"]:
        price_name = f"last60_{side}_price_1"
        last60_columns.extend(f"{price_name}_{suffix}" for suffix in ["mean", "std", "start", "end"])
    last60_columns.extend(["last60_ask_volume_1_mean", "last60_bid_volume_1_mean", "last60_spread_1_mean", "last60_mid_price_1_change", "last60_book_volume_imbalance_1"])
    level2_columns = ["sample_id"]
    for side in ["ask", "bid"]:
        price_name = f"{side}_price_2"
        volume_name = f"{side}_volume_2"
        level2_columns.extend(f"{price_name}_{suffix}" for suffix in ["mean", "std", "min", "max", "first", "last"])
        level2_columns.extend(f"{volume_name}_{suffix}" for suffix in ["sum", "mean", "max"])
        level2_columns.extend(f"last60_{price_name}_{suffix}" for suffix in ["mean", "std", "start", "end"])
        level2_columns.append(f"last60_{volume_name}_mean")
    level2_columns.extend([
        "spread_2_mean", "mid_price_2_mean", "mid_price_2_change", "book_volume_imbalance_2",
        "last60_spread_2_mean", "last60_mid_price_2_change", "last60_book_volume_imbalance_2",
    ])
    return data.select(v1_columns), data.select(last60_columns), data.select(level2_columns)


def main() -> None:
    """Build a complete split directly; no smoke mode is provided."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    market_path = project_dir / "data" / "raw" / arguments.split / "market.feather"
    full_data = build_aggregate_table(market_path)
    v1, last60, level2 = derive_and_split(full_data)
    processed_dir = project_dir / "data" / "processed"
    safe_write(v1, processed_dir / f"{arguments.split}_market_features.feather")
    safe_write(last60, processed_dir / f"{arguments.split}_market_last60_features_complete.feather")
    safe_write(level2, processed_dir / f"{arguments.split}_market_level2_features.feather")
    print(f"V1 shape = {v1.shape}; last60 shape = {last60.shape}; L2 shape = {level2.shape}")


if __name__ == "__main__":
    main()
