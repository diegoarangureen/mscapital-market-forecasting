"""Build one-row-per-sample market features with bounded peak memory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


# 每次子进程只读取 sample_id 和一个特征列，避免完整宽表进入内存。
# Each child process reads sample_id plus one feature column to avoid loading the full wide table.
FEATURE_COLUMNS = [
    "seconds_before_predict",
    "transaction_count",
    "transaction_volume",
    "transaction_avgprice",
    "ask_price_1",
    "bid_price_1",
    "ask_volume_1",
    "bid_volume_1",
]


def build_aggregation_expressions(column_name: str) -> list[pl.Expr]:
    """Return the fixed aggregate expressions for one source column."""

    column = pl.col(column_name)

    # 时间列同时负责样本长度、覆盖范围和长空档检查。
    # The time column also provides sample length, coverage, and long-gap checks.
    if column_name == "seconds_before_predict":
        return [
            pl.len().alias("market_row_count"),
            column.max().alias("seconds_max"),
            column.min().alias("seconds_min"),
            (column.diff().abs() > 4.5).sum().alias("long_gap_count"),
        ]

    # 成交次数提供总强度、典型强度、峰值和有成交时间段比例。
    # Transaction count provides total, typical, peak, and active-period ratio features.
    if column_name == "transaction_count":
        return [
            column.sum().alias("transaction_count_sum"),
            column.mean().alias("transaction_count_mean"),
            column.max().alias("transaction_count_max"),
            (column > 0).mean().alias("has_transaction_ratio"),
        ]

    # 成交量保留总量、平均水平和单个时间段峰值。
    # Transaction volume keeps total, average, and single-period peak activity.
    if column_name == "transaction_volume":
        return [
            column.sum().alias("transaction_volume_sum"),
            column.mean().alias("transaction_volume_mean"),
            column.max().alias("transaction_volume_max"),
        ]

    # 平均成交价允许 null；均值和标准差会忽略无成交时间段。
    # Average transaction price may be null; mean and standard deviation ignore inactive periods.
    if column_name == "transaction_avgprice":
        return [
            column.mean().alias("transaction_avgprice_mean"),
            column.std().alias("transaction_avgprice_std"),
            column.min().alias("transaction_avgprice_min"),
            column.max().alias("transaction_avgprice_max"),
            column.drop_nulls().first().alias("transaction_avgprice_first_valid"),
            column.drop_nulls().last().alias("transaction_avgprice_last_valid"),
        ]

    # 第一档买卖价保留水平、波动、范围和首尾状态。
    # Level-one bid and ask prices keep level, volatility, range, and endpoints.
    if column_name in {"ask_price_1", "bid_price_1"}:
        return [
            column.mean().alias(f"{column_name}_mean"),
            column.std().alias(f"{column_name}_std"),
            column.min().alias(f"{column_name}_min"),
            column.max().alias(f"{column_name}_max"),
            column.first().alias(f"{column_name}_first"),
            column.last().alias(f"{column_name}_last"),
        ]

    # 第一档买卖量保留窗口总量、平均深度和峰值深度。
    # Level-one bid and ask volumes keep total, average, and peak depth.
    if column_name in {"ask_volume_1", "bid_volume_1"}:
        return [
            column.sum().alias(f"{column_name}_sum"),
            column.mean().alias(f"{column_name}_mean"),
            column.max().alias(f"{column_name}_max"),
        ]

    raise ValueError(f"No aggregation plan for column: {column_name}")


def aggregate_one_column(market_path: Path, column_name: str, output_path: Path) -> None:
    """Aggregate one full market column in an isolated process."""

    # 只投影两列；子进程结束后约 3 GiB 的峰值内存会被操作系统释放。
    # Project only two columns; the operating system releases peak memory when this process exits.
    column_data = pl.read_ipc(
        market_path,
        columns=["sample_id", column_name],
        memory_map=False,
    )

    # 原文件按 sample_id 和倒计时顺序保存；maintain_order 保留组首次出现的顺序。
    # The source is ordered by sample ID and countdown; maintain_order preserves first group order.
    aggregated_data = column_data.group_by(
        "sample_id",
        maintain_order=True,
    ).agg(build_aggregation_expressions(column_name))

    # 每个中间文件已经从 2.217 亿行缩小到约 125 万行。
    # Each intermediate file is reduced from 221.7 million rows to about 1.25 million rows.
    aggregated_data.write_ipc(output_path)

    print(
        f"finished column={column_name}, "
        f"input_shape={column_data.shape}, output_shape={aggregated_data.shape}"
    )


def combine_feature_parts(part_paths: list[Path], output_path: Path) -> None:
    """Join compact per-column feature parts and add derived financial features."""

    # 第一个部件建立 sample_id 主表，其余部件必须一对一对齐。
    # The first part establishes the sample-ID table; every later part must join one-to-one.
    combined_features = pl.read_ipc(part_paths[0])

    for part_path in part_paths[1:]:
        feature_part = pl.read_ipc(part_path)
        combined_features = combined_features.join(
            feature_part,
            on="sample_id",
            how="inner",
            validate="1:1",
        )

    # 根据已经聚合的买卖价，计算平均价差和平均中间价。
    # Derive average spread and mid-price from the aggregated bid and ask prices.
    combined_features = combined_features.with_columns(
        (pl.col("ask_price_1_mean") - pl.col("bid_price_1_mean")).alias(
            "spread_1_mean"
        ),
        (
            (pl.col("ask_price_1_mean") + pl.col("bid_price_1_mean"))
            / 2.0
        ).alias("mid_price_1_mean"),
        (pl.col("seconds_max") - pl.col("seconds_min")).alias("seconds_coverage"),
    )

    # 首尾中间价之差描述约 600 秒窗口中的净价格变化。
    # The endpoint mid-price difference describes net movement over the roughly 600-second window.
    combined_features = combined_features.with_columns(
        (
            (
                pl.col("ask_price_1_last")
                + pl.col("bid_price_1_last")
                - pl.col("ask_price_1_first")
                - pl.col("bid_price_1_first")
            )
            / 2.0
        ).alias("mid_price_1_change"),
    )

    # 汇总买卖盘总量形成 [-1, 1] 附近的方向性不平衡特征。
    # Convert total bid and ask depth into a directional imbalance feature near [-1, 1].
    volume_denominator = pl.col("bid_volume_1_sum") + pl.col("ask_volume_1_sum")
    combined_features = combined_features.with_columns(
        (
            (pl.col("bid_volume_1_sum") - pl.col("ask_volume_1_sum"))
            / volume_denominator
        ).alias("book_volume_imbalance_1")
    )

    # 保存可直接与 label 按 sample_id 对齐的完整训练特征表。
    # Save the complete training feature table ready to align with labels by sample_id.
    combined_features.write_ipc(output_path)

    print(f"combined feature shape = {combined_features.shape}")
    print(f"combined feature output = {output_path}")


def build_full_train_features(project_dir: Path) -> None:
    """Run isolated column aggregations sequentially and combine the results."""

    market_path = project_dir / "data" / "raw" / "train" / "market.feather"
    part_dir = project_dir / "data" / "interim" / "train_market_feature_parts"
    output_path = project_dir / "data" / "processed" / "train_market_features.feather"

    # 创建项目内的中间与最终目录；不会改动 data/raw。
    # Create project-local interim and output directories without modifying data/raw.
    part_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []

    for column_name in FEATURE_COLUMNS:
        part_path = part_dir / f"{column_name}_features.feather"
        part_paths.append(part_path)

        # 已完成的中间列可以复用，意外中断后不必从头再算。
        # Reuse completed parts so an interrupted build does not restart from zero.
        if part_path.exists():
            print(f"reusing column={column_name}, path={part_path}")
            continue

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--aggregate-column",
            "--market-path",
            str(market_path),
            "--column-name",
            column_name,
            "--output-path",
            str(part_path),
        ]

        # check=True 会在任何单列失败时停止，避免合并不完整结果。
        # check=True stops on any failed column to avoid combining incomplete results.
        subprocess.run(command, check=True)

    combine_feature_parts(part_paths, output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse driver mode and isolated child-process mode arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-column", action="store_true")
    parser.add_argument("--market-path", type=Path)
    parser.add_argument("--column-name")
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.aggregate_column:
        aggregate_one_column(
            arguments.market_path,
            arguments.column_name,
            arguments.output_path,
        )
    else:
        current_project_dir = Path(__file__).resolve().parents[1]
        build_full_train_features(current_project_dir)
