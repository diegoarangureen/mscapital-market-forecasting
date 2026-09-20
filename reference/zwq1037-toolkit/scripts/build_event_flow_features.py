"""Build target-free transaction and order event-flow features at full scale.

The raw event files contain one very large Arrow record batch.  Polars' lazy
streaming engine projects only the required columns and performs one grouped
aggregation per source.  Train and test use the same formulas.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import polars as pl


TRANSACTION_FEATURE_COLUMNS = [
    "trade_volume_imbalance_60",
    "trade_volume_imbalance_20",
    "trade_count_imbalance_60",
    "trade_average_size_imbalance_60",
    "trade_pressure_acceleration_20_vs_40",
    "log1p_total_trade_volume_60",
    "log1p_total_trade_count_60",
    "last_trade_seconds_before_predict",
]

ORDER_FEATURE_COLUMNS = [
    "new_order_volume_imbalance_60",
    "cancel_order_pressure_60",
    "net_order_pressure_60",
    "buy_cancel_ratio_60",
    "sell_cancel_ratio_60",
    "total_cancel_ratio_60",
    "net_order_pressure_20",
    "net_order_pressure_acceleration_20_vs_40",
]


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a ratio only when its denominator is strictly positive."""

    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


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


def conditional_volume(condition: pl.Expr, alias: str) -> pl.Expr:
    """Sum event volume under one direction/window condition using 64-bit storage."""

    return (
        pl.when(condition)
        .then(pl.col("volume").cast(pl.Int64))
        .otherwise(0)
        .sum()
        .alias(alias)
    )


def conditional_count(condition: pl.Expr, alias: str) -> pl.Expr:
    """Count events under one direction/window condition."""

    return condition.cast(pl.Int64).sum().alias(alias)


def master_sample_ids(project_dir: Path, split: str) -> pl.DataFrame:
    """Load the complete expected sample-ID list without reading labels into features."""

    if split == "train":
        return pl.read_ipc(
            project_dir / "data" / "raw" / "label.feather",
            columns=["sample_id"],
        )
    return pl.read_csv(
        project_dir / "data" / "raw" / "submission.csv",
        columns=["sample_id"],
    )


def build_transaction_features(project_dir: Path, split: str) -> Path:
    """Aggregate active-buy and active-sell transaction pressure."""

    input_path = project_dir / "data" / "raw" / split / "transaction.feather"
    output_path = (
        project_dir / "data" / "processed" / f"{split}_transaction_flow_features.feather"
    )
    seconds = pl.col("seconds_before_predict")
    side = pl.col("side")
    in_60 = seconds <= 60.0
    in_20 = seconds <= 20.0
    in_previous_40 = (seconds > 20.0) & (seconds <= 60.0)
    buy_60 = in_60 & (side == 0)
    sell_60 = in_60 & (side == 1)
    buy_20 = in_20 & (side == 0)
    sell_20 = in_20 & (side == 1)
    buy_previous_40 = in_previous_40 & (side == 0)
    sell_previous_40 = in_previous_40 & (side == 1)

    aggregates = (
        pl.scan_ipc(input_path)
        .select(["sample_id", "seconds_before_predict", "volume", "side"])
        .group_by("sample_id")
        .agg(
            conditional_volume(buy_60, "buy_volume_60"),
            conditional_volume(sell_60, "sell_volume_60"),
            conditional_volume(buy_20, "buy_volume_20"),
            conditional_volume(sell_20, "sell_volume_20"),
            conditional_volume(buy_previous_40, "buy_volume_previous_40"),
            conditional_volume(sell_previous_40, "sell_volume_previous_40"),
            conditional_count(buy_60, "buy_count_60"),
            conditional_count(sell_60, "sell_count_60"),
            pl.col("seconds_before_predict").filter(in_60).min().alias("last_trade_seconds_before_predict"),
        )
        .collect(engine="streaming")
    )
    aggregates = master_sample_ids(project_dir, split).join(
        aggregates, on="sample_id", how="left", validate="1:1"
    )
    zero_columns = [
        "buy_volume_60",
        "sell_volume_60",
        "buy_volume_20",
        "sell_volume_20",
        "buy_volume_previous_40",
        "sell_volume_previous_40",
        "buy_count_60",
        "sell_count_60",
    ]
    aggregates = aggregates.with_columns(pl.col(zero_columns).fill_null(0))
    total_volume_60 = pl.col("buy_volume_60") + pl.col("sell_volume_60")
    total_volume_20 = pl.col("buy_volume_20") + pl.col("sell_volume_20")
    total_volume_previous_40 = (
        pl.col("buy_volume_previous_40") + pl.col("sell_volume_previous_40")
    )
    total_count_60 = pl.col("buy_count_60") + pl.col("sell_count_60")
    buy_average_size = safe_ratio(pl.col("buy_volume_60"), pl.col("buy_count_60"))
    sell_average_size = safe_ratio(pl.col("sell_volume_60"), pl.col("sell_count_60"))
    transaction_imbalance_20 = safe_ratio(
        pl.col("buy_volume_20") - pl.col("sell_volume_20"), total_volume_20
    )
    transaction_imbalance_previous_40 = safe_ratio(
        pl.col("buy_volume_previous_40") - pl.col("sell_volume_previous_40"),
        total_volume_previous_40,
    )
    features = aggregates.with_columns(
        safe_ratio(
            pl.col("buy_volume_60") - pl.col("sell_volume_60"), total_volume_60
        ).alias("trade_volume_imbalance_60"),
        transaction_imbalance_20.alias("trade_volume_imbalance_20"),
        safe_ratio(
            pl.col("buy_count_60") - pl.col("sell_count_60"), total_count_60
        ).alias("trade_count_imbalance_60"),
        safe_ratio(
            buy_average_size - sell_average_size,
            buy_average_size + sell_average_size,
        ).alias("trade_average_size_imbalance_60"),
        (transaction_imbalance_20 - transaction_imbalance_previous_40).alias(
            "trade_pressure_acceleration_20_vs_40"
        ),
        total_volume_60.cast(pl.Float64).log1p().alias("log1p_total_trade_volume_60"),
        total_count_60.cast(pl.Float64).log1p().alias("log1p_total_trade_count_60"),
    ).select("sample_id", *TRANSACTION_FEATURE_COLUMNS)
    safe_write_ipc(features, output_path)
    print(f"transaction feature shape = {features.shape}", flush=True)
    print(f"transaction output = {output_path}", flush=True)
    return output_path


def build_order_features(project_dir: Path, split: str) -> Path:
    """Aggregate new/cancel buy/sell order-event pressure."""

    input_path = project_dir / "data" / "raw" / split / "order.feather"
    output_path = project_dir / "data" / "processed" / f"{split}_order_flow_features.feather"
    seconds = pl.col("seconds_before_predict")
    side = pl.col("side")
    action = pl.col("order_action")
    in_60 = seconds <= 60.0
    in_20 = seconds <= 20.0
    in_previous_40 = (seconds > 20.0) & (seconds <= 60.0)

    def order_condition(window: pl.Expr, side_code: int, action_code: int) -> pl.Expr:
        return window & (side == side_code) & (action == action_code)

    aggregate_expressions = []
    for window_name, window in [
        ("60", in_60),
        ("20", in_20),
        ("previous_40", in_previous_40),
    ]:
        for side_name, side_code in [("buy", 0), ("sell", 1)]:
            for action_name, action_code in [("new", 0), ("cancel", 1)]:
                alias = f"{action_name}_{side_name}_volume_{window_name}"
                aggregate_expressions.append(
                    conditional_volume(
                        order_condition(window, side_code, action_code), alias
                    )
                )

    aggregates = (
        pl.scan_ipc(input_path)
        .select(
            [
                "sample_id",
                "seconds_before_predict",
                "volume",
                "side",
                "order_action",
            ]
        )
        .group_by("sample_id")
        .agg(aggregate_expressions)
        .collect(engine="streaming")
    )
    aggregates = master_sample_ids(project_dir, split).join(
        aggregates, on="sample_id", how="left", validate="1:1"
    )
    aggregate_columns = [column for column in aggregates.columns if column != "sample_id"]
    aggregates = aggregates.with_columns(pl.col(aggregate_columns).fill_null(0))

    def net_pressure(window_name: str) -> pl.Expr:
        new_buy = pl.col(f"new_buy_volume_{window_name}")
        new_sell = pl.col(f"new_sell_volume_{window_name}")
        cancel_buy = pl.col(f"cancel_buy_volume_{window_name}")
        cancel_sell = pl.col(f"cancel_sell_volume_{window_name}")
        denominator = new_buy + new_sell + cancel_buy + cancel_sell
        return safe_ratio(new_buy - new_sell - cancel_buy + cancel_sell, denominator)

    new_buy_60 = pl.col("new_buy_volume_60")
    new_sell_60 = pl.col("new_sell_volume_60")
    cancel_buy_60 = pl.col("cancel_buy_volume_60")
    cancel_sell_60 = pl.col("cancel_sell_volume_60")
    total_order_volume_60 = new_buy_60 + new_sell_60 + cancel_buy_60 + cancel_sell_60
    features = aggregates.with_columns(
        safe_ratio(new_buy_60 - new_sell_60, new_buy_60 + new_sell_60).alias(
            "new_order_volume_imbalance_60"
        ),
        safe_ratio(cancel_sell_60 - cancel_buy_60, cancel_buy_60 + cancel_sell_60).alias(
            "cancel_order_pressure_60"
        ),
        net_pressure("60").alias("net_order_pressure_60"),
        safe_ratio(cancel_buy_60, new_buy_60 + cancel_buy_60).alias("buy_cancel_ratio_60"),
        safe_ratio(cancel_sell_60, new_sell_60 + cancel_sell_60).alias("sell_cancel_ratio_60"),
        safe_ratio(cancel_buy_60 + cancel_sell_60, total_order_volume_60).alias(
            "total_cancel_ratio_60"
        ),
        net_pressure("20").alias("net_order_pressure_20"),
        (net_pressure("20") - net_pressure("previous_40")).alias(
            "net_order_pressure_acceleration_20_vs_40"
        ),
    ).select("sample_id", *ORDER_FEATURE_COLUMNS)
    safe_write_ipc(features, output_path)
    print(f"order feature shape = {features.shape}", flush=True)
    print(f"order output = {output_path}", flush=True)
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Choose the data split and full source to build; no smoke mode is provided."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--source", choices=["transaction", "order", "all"], default="all"
    )
    return parser.parse_args()


def main() -> None:
    """Run full-scale feature aggregation for the explicitly selected source."""

    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parents[1]
    if arguments.source in {"transaction", "all"}:
        build_transaction_features(project_dir, arguments.split)
    if arguments.source in {"order", "all"}:
        build_order_features(project_dir, arguments.split)


if __name__ == "__main__":
    main()
