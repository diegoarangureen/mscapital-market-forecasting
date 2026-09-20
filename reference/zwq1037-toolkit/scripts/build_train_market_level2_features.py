"""Build full-window and last-60-second level-two market features safely."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl


# 每个子进程只读取ID、倒计时和一个第二档字段，限制峰值内存。
# Each child reads ID, countdown, and one level-two field to bound peak memory.
SOURCE_COLUMNS = [
    "ask_price_2",
    "bid_price_2",
    "ask_volume_2",
    "bid_volume_2",
]


def full_expressions(column_name: str) -> list[pl.Expr]:
    """Return full-window aggregates that mirror the existing level-one features."""

    column = pl.col(column_name)
    if column_name in {"ask_price_2", "bid_price_2"}:
        return [
            column.mean().alias(f"{column_name}_mean"),
            column.std().alias(f"{column_name}_std"),
            column.min().alias(f"{column_name}_min"),
            column.max().alias(f"{column_name}_max"),
            column.first().alias(f"{column_name}_first"),
            column.last().alias(f"{column_name}_last"),
        ]
    if column_name in {"ask_volume_2", "bid_volume_2"}:
        return [
            column.sum().alias(f"{column_name}_sum"),
            column.mean().alias(f"{column_name}_mean"),
            column.max().alias(f"{column_name}_max"),
        ]
    raise ValueError(f"No full-window plan for {column_name}.")


def last60_expressions(column_name: str) -> list[pl.Expr]:
    """Return recent-window aggregates matching the existing level-one design."""

    column = pl.col(column_name)
    if column_name in {"ask_price_2", "bid_price_2"}:
        return [
            column.mean().alias(f"last60_{column_name}_mean"),
            column.std().alias(f"last60_{column_name}_std"),
            column.first().alias(f"last60_{column_name}_start"),
            column.last().alias(f"last60_{column_name}_end"),
        ]
    if column_name in {"ask_volume_2", "bid_volume_2"}:
        return [column.mean().alias(f"last60_{column_name}_mean")]
    raise ValueError(f"No last-60-second plan for {column_name}.")


def safe_write_ipc(data: pl.DataFrame, output_path: Path) -> None:
    """Write through an ASCII temporary path for reliable Windows Unicode handling."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        data.write_ipc(temporary_path)
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def aggregate_one_column(
    market_path: Path,
    column_name: str,
    output_path: Path,
) -> None:
    """Aggregate one level-two field over both windows in an isolated process."""

    # 同一次读取同时生成全段和近期统计，避免对2.2亿行重复扫描两次。
    # One read generates both windows, avoiding two scans of 221.7 million rows.
    column_data = pl.read_ipc(
        market_path,
        columns=["sample_id", "seconds_before_predict", column_name],
        memory_map=False,
    )
    full_features = column_data.group_by("sample_id", maintain_order=True).agg(
        full_expressions(column_name)
    )
    recent_features = (
        column_data.filter(pl.col("seconds_before_predict") <= 60.0)
        .group_by("sample_id", maintain_order=True)
        .agg(last60_expressions(column_name))
    )

    # 全段sample_id是主表；没有最后60秒记录的样本保留，近期统计为null。
    # Full-window IDs are the master; missing recent windows remain as null statistics.
    combined_column = full_features.join(
        recent_features,
        on="sample_id",
        how="left",
        validate="1:1",
    )
    safe_write_ipc(combined_column, output_path)
    print(
        f"finished column={column_name}, input_shape={column_data.shape}, "
        f"output_shape={combined_column.shape}"
    )


def combine_parts(part_paths: list[Path], output_path: Path) -> None:
    """Join the four compact parts and derive level-two book features."""

    combined_features = pl.read_ipc(part_paths[0])
    for part_path in part_paths[1:]:
        combined_features = combined_features.join(
            pl.read_ipc(part_path),
            on="sample_id",
            how="inner",
            validate="1:1",
        )

    full_volume_denominator = (
        pl.col("bid_volume_2_sum") + pl.col("ask_volume_2_sum")
    )
    recent_volume_denominator = (
        pl.col("last60_bid_volume_2_mean")
        + pl.col("last60_ask_volume_2_mean")
    )
    combined_features = combined_features.with_columns(
        (pl.col("ask_price_2_mean") - pl.col("bid_price_2_mean")).alias(
            "spread_2_mean"
        ),
        ((pl.col("ask_price_2_mean") + pl.col("bid_price_2_mean")) / 2.0).alias(
            "mid_price_2_mean"
        ),
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
        .then(
            (pl.col("bid_volume_2_sum") - pl.col("ask_volume_2_sum"))
            / full_volume_denominator
        )
        .otherwise(None)
        .alias("book_volume_imbalance_2"),
        (
            pl.col("last60_ask_price_2_mean")
            - pl.col("last60_bid_price_2_mean")
        ).alias("last60_spread_2_mean"),
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
            (
                pl.col("last60_bid_volume_2_mean")
                - pl.col("last60_ask_volume_2_mean")
            )
            / recent_volume_denominator
        )
        .otherwise(None)
        .alias("last60_book_volume_imbalance_2"),
    )
    if combined_features.width != 36:
        raise AssertionError(
            f"Expected sample_id plus 35 features, got width={combined_features.width}."
        )
    safe_write_ipc(combined_features, output_path)
    print(f"combined level-two feature shape = {combined_features.shape}")
    print(f"combined level-two output = {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse driver and isolated child-process arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-column", action="store_true")
    parser.add_argument("--market-path", type=Path)
    parser.add_argument("--column-name", choices=SOURCE_COLUMNS)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--part-dir", type=Path)
    return parser.parse_args()


def run_driver(arguments: argparse.Namespace) -> None:
    """Build or reuse four isolated parts, then combine them."""

    project_dir = Path(__file__).resolve().parents[1]
    market_path = arguments.market_path or (
        project_dir / "data" / "raw" / "train" / "market.feather"
    )
    part_dir = arguments.part_dir or (
        project_dir / "data" / "interim" / "train_market_level2_feature_parts"
    )
    output_path = arguments.output_path or (
        project_dir / "data" / "processed" / "train_market_level2_features.feather"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []
    for column_name in SOURCE_COLUMNS:
        part_path = part_dir / f"{column_name}_features.feather"
        part_paths.append(part_path)
        if part_path.exists():
            print(f"reusing column={column_name}, path={part_path}")
            continue
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--aggregate-column",
                "--market-path",
                str(market_path),
                "--column-name",
                column_name,
                "--output-path",
                str(part_path),
            ],
            check=True,
        )
    combine_parts(part_paths, output_path)


if __name__ == "__main__":
    parsed_arguments = parse_arguments()
    if parsed_arguments.aggregate_column:
        aggregate_one_column(
            parsed_arguments.market_path,
            parsed_arguments.column_name,
            parsed_arguments.output_path,
        )
    else:
        run_driver(parsed_arguments)
