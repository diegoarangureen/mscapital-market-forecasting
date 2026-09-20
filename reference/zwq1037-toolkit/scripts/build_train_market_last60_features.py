"""Build last-60-second market features with bounded peak memory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl


# 每次子进程只处理一个源特征，避免完整宽表进入内存。
# Each child process handles one source feature to avoid loading the full wide table.
SOURCE_COLUMNS = [
    "seconds_before_predict",
    "transaction_count",
    "transaction_volume",
    "transaction_avgprice",
    "ask_price_1",
    "bid_price_1",
    "ask_volume_1",
    "bid_volume_1",
]


def build_last60_expressions(column_name: str) -> list[pl.Expr]:
    """Return last-60-second aggregates for one source column."""

    column = pl.col(column_name)

    if column_name == "seconds_before_predict":
        return [pl.len().alias("last60_row_count")]

    if column_name == "transaction_count":
        return [
            column.sum().alias("last60_transaction_count_sum"),
            column.max().alias("last60_transaction_count_max"),
            (column > 0).mean().alias("last60_has_transaction_ratio"),
        ]

    if column_name == "transaction_volume":
        return [
            column.sum().alias("last60_transaction_volume_sum"),
            column.max().alias("last60_transaction_volume_max"),
        ]

    if column_name == "transaction_avgprice":
        return [
            column.mean().alias("last60_transaction_avgprice_mean"),
            column.std().alias("last60_transaction_avgprice_std"),
            column.drop_nulls().last().alias(
                "last60_transaction_avgprice_last_valid"
            ),
        ]

    if column_name in {"ask_price_1", "bid_price_1"}:
        return [
            column.mean().alias(f"last60_{column_name}_mean"),
            column.std().alias(f"last60_{column_name}_std"),
            column.first().alias(f"last60_{column_name}_start"),
            column.last().alias(f"last60_{column_name}_end"),
        ]

    if column_name in {"ask_volume_1", "bid_volume_1"}:
        return [column.mean().alias(f"last60_{column_name}_mean")]

    raise ValueError(f"No last-60-second aggregation plan for: {column_name}")


def aggregate_one_column(
    market_path: Path,
    column_name: str,
    output_path: Path,
) -> None:
    """Filter and aggregate one source column in an isolated process."""

    # 时间列本身只需读取两列；其他特征还需要时间列来建立窗口掩码。
    # The time column needs two inputs; other features also need time for the window mask.
    read_columns = ["sample_id", "seconds_before_predict"]
    if column_name != "seconds_before_predict":
        read_columns.append(column_name)

    column_data = pl.read_ipc(
        market_path,
        columns=read_columns,
        memory_map=False,
    )

    # 倒计时越小越接近预测时点；先过滤可显著减少后续聚合的数据量。
    # Smaller countdowns are closer to prediction; filtering first reduces aggregation work.
    recent_data = column_data.filter(
        pl.col("seconds_before_predict") <= 60.0
    )

    aggregated_data = recent_data.group_by(
        "sample_id",
        maintain_order=True,
    ).agg(build_last60_expressions(column_name))

    aggregated_data.write_ipc(output_path)

    print(
        f"finished column={column_name}, "
        f"full_shape={column_data.shape}, "
        f"recent_shape={recent_data.shape}, "
        f"output_shape={aggregated_data.shape}"
    )


def combine_parts(
    part_paths: list[Path],
    output_path: Path,
    master_sample_path: Path | None = None,
) -> None:
    """Join compact parts and add derived recent-window features."""

    # 完整构建时以V1全部sample_id为主表，保留最后60秒没有记录的样本。
    # In the full build, use every V1 sample ID as the master to retain samples with no last-60 rows.
    if master_sample_path is None:
        combined_features = pl.read_ipc(part_paths[0])
        remaining_part_paths = part_paths[1:]
    else:
        combined_features = pl.read_ipc(
            master_sample_path,
            columns=["sample_id"],
        )
        remaining_part_paths = part_paths

    for part_path in remaining_part_paths:
        feature_part = pl.read_ipc(part_path)
        combined_features = combined_features.join(
            feature_part,
            on="sample_id",
            how="left",
            validate="1:1",
        )

    # 没有任何最后60秒记录时，行数事实为0；其他统计量保持null。
    # When no last-60 rows exist, the factual row count is zero; other statistics remain null.
    combined_features = combined_features.with_columns(
        pl.col("last60_row_count").fill_null(0)
    )

    # 近期平均价差与中间价变化把窗口含义编码为普通数值列。
    # Recent mean spread and mid-price change encode window meaning as numeric columns.
    combined_features = combined_features.with_columns(
        (
            pl.col("last60_ask_price_1_mean")
            - pl.col("last60_bid_price_1_mean")
        ).alias("last60_spread_1_mean"),
        (
            (
                pl.col("last60_ask_price_1_end")
                + pl.col("last60_bid_price_1_end")
                - pl.col("last60_ask_price_1_start")
                - pl.col("last60_bid_price_1_start")
            )
            / 2.0
        ).alias("last60_mid_price_1_change"),
    )

    # 买卖盘平均深度形成最后60秒的方向性不平衡特征。
    # Mean bid/ask depth forms a directional imbalance feature for the final 60 seconds.
    volume_denominator = (
        pl.col("last60_bid_volume_1_mean")
        + pl.col("last60_ask_volume_1_mean")
    )
    combined_features = combined_features.with_columns(
        (
            (
                pl.col("last60_bid_volume_1_mean")
                - pl.col("last60_ask_volume_1_mean")
            )
            / volume_denominator
        ).alias("last60_book_volume_imbalance_1")
    )

    # Polars的底层写入器在部分Windows环境无法处理中文路径；先写英文临时路径。
    # Polars' native writer may reject Unicode paths on Windows; write to an ASCII temp path first.
    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".feather",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        combined_features.write_ipc(temporary_path)

        # Python文件接口可以安全复制到当前中文项目路径。
        # Python's file interface can safely copy the result into the Unicode project path.
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"combined last60 feature shape = {combined_features.shape}")
    print(f"combined last60 feature output = {output_path}")


def build_full_features(project_dir: Path) -> None:
    """Run isolated aggregations sequentially and combine their outputs."""

    market_path = project_dir / "data" / "raw" / "train" / "market.feather"
    part_dir = (
        project_dir
        / "data"
        / "interim"
        / "train_market_last60_feature_parts"
    )
    output_path = (
        project_dir
        / "data"
        / "processed"
        / "train_market_last60_features_complete.feather"
    )

    part_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []

    for column_name in SOURCE_COLUMNS:
        part_path = part_dir / f"{column_name}_last60_features.feather"
        part_paths.append(part_path)

        # 已完成的中间结果可复用，构建中断后无需从头开始。
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
        subprocess.run(command, check=True)

    master_sample_path = (
        project_dir
        / "data"
        / "processed"
        / "train_market_features.feather"
    )
    combine_parts(
        part_paths,
        output_path,
        master_sample_path=master_sample_path,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse driver and isolated child-process arguments."""

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
        build_full_features(current_project_dir)
