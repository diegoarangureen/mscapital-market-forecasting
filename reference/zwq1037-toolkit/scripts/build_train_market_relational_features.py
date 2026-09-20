"""Build target-free temporal and cross-level relational market features."""

import shutil
import tempfile
from pathlib import Path

import polars as pl


TEMPORAL_FEATURE_COLUMNS = [
    "recent_row_ratio",
    "recent_transaction_count_share",
    "recent_transaction_volume_share",
    "recent_has_transaction_ratio_delta",
    "recent_transaction_avgprice_delta",
    "recent_spread_1_delta",
    "recent_mid_price_1_change_delta",
    "recent_book_imbalance_1_delta",
    "recent_spread_2_delta",
    "recent_mid_price_2_change_delta",
    "recent_book_imbalance_2_delta",
]

CROSS_LEVEL_FEATURE_COLUMNS = [
    "level2_spread_minus_level1_spread",
    "level2_mid_minus_level1_mid",
    "level2_mid_change_minus_level1_mid_change",
    "level2_imbalance_minus_level1_imbalance",
    "ask_volume_2_to_1_sum_ratio",
    "bid_volume_2_to_1_sum_ratio",
    "total_volume_2_to_1_sum_ratio",
    "last60_level2_spread_minus_level1_spread",
    "last60_level2_mid_change_minus_level1_mid_change",
    "last60_level2_imbalance_minus_level1_imbalance",
    "last60_total_volume_2_to_1_mean_ratio",
]


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a ratio, preserving null when the denominator is zero or unavailable."""

    return (
        pl.when(denominator.is_not_null() & (denominator.abs() > 1e-12))
        .then(numerator / denominator)
        .otherwise(None)
    )


def safe_write_ipc(data: pl.DataFrame, output_path: Path) -> None:
    """Write through an ASCII temporary path for Windows Unicode reliability."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        data.write_ipc(temporary_path)
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    project_dir = Path(__file__).resolve().parents[1]
    v1_path = project_dir / "data" / "processed" / "train_market_features.feather"
    last60_path = (
        project_dir
        / "data"
        / "processed"
        / "train_market_last60_features_complete.feather"
    )
    level2_path = (
        project_dir / "data" / "processed" / "train_market_level2_features.feather"
    )
    output_path = (
        project_dir
        / "data"
        / "processed"
        / "train_market_relational_features.feather"
    )

    # 只读取推导公式需要的列；三张表都以唯一sample_id一对一连接。
    # Read only required columns; all three tables join one-to-one by sample ID.
    v1 = pl.read_ipc(
        v1_path,
        columns=[
            "sample_id",
            "market_row_count",
            "transaction_count_sum",
            "transaction_volume_sum",
            "has_transaction_ratio",
            "transaction_avgprice_mean",
            "spread_1_mean",
            "mid_price_1_mean",
            "mid_price_1_change",
            "book_volume_imbalance_1",
            "ask_volume_1_sum",
            "bid_volume_1_sum",
        ],
    )
    last60 = pl.read_ipc(
        last60_path,
        columns=[
            "sample_id",
            "last60_row_count",
            "last60_transaction_count_sum",
            "last60_transaction_volume_sum",
            "last60_has_transaction_ratio",
            "last60_transaction_avgprice_mean",
            "last60_spread_1_mean",
            "last60_mid_price_1_change",
            "last60_book_volume_imbalance_1",
            "last60_ask_volume_1_mean",
            "last60_bid_volume_1_mean",
        ],
    )
    level2 = pl.read_ipc(
        level2_path,
        columns=[
            "sample_id",
            "spread_2_mean",
            "mid_price_2_mean",
            "mid_price_2_change",
            "book_volume_imbalance_2",
            "ask_volume_2_sum",
            "bid_volume_2_sum",
            "last60_spread_2_mean",
            "last60_mid_price_2_change",
            "last60_book_volume_imbalance_2",
            "last60_ask_volume_2_mean",
            "last60_bid_volume_2_mean",
        ],
    )
    source = (
        v1.join(last60, on="sample_id", how="left", validate="1:1")
        .join(level2, on="sample_id", how="left", validate="1:1")
        .with_columns(pl.col("last60_row_count").fill_null(0))
    )

    level1_total_volume = pl.col("ask_volume_1_sum") + pl.col("bid_volume_1_sum")
    level2_total_volume = pl.col("ask_volume_2_sum") + pl.col("bid_volume_2_sum")
    recent_level1_total_volume = (
        pl.col("last60_ask_volume_1_mean")
        + pl.col("last60_bid_volume_1_mean")
    )
    recent_level2_total_volume = (
        pl.col("last60_ask_volume_2_mean")
        + pl.col("last60_bid_volume_2_mean")
    )

    # 近期相对全段：把“最近状态是否偏离过去”直接交给树。
    # Recent-versus-full features expose whether the latest state departs from history.
    temporal_expressions = [
        safe_ratio(pl.col("last60_row_count"), pl.col("market_row_count")).alias(
            "recent_row_ratio"
        ),
        safe_ratio(
            pl.col("last60_transaction_count_sum"),
            pl.col("transaction_count_sum"),
        ).alias("recent_transaction_count_share"),
        safe_ratio(
            pl.col("last60_transaction_volume_sum"),
            pl.col("transaction_volume_sum"),
        ).alias("recent_transaction_volume_share"),
        (
            pl.col("last60_has_transaction_ratio")
            - pl.col("has_transaction_ratio")
        ).alias("recent_has_transaction_ratio_delta"),
        (
            pl.col("last60_transaction_avgprice_mean")
            - pl.col("transaction_avgprice_mean")
        ).alias("recent_transaction_avgprice_delta"),
        (pl.col("last60_spread_1_mean") - pl.col("spread_1_mean")).alias(
            "recent_spread_1_delta"
        ),
        (
            pl.col("last60_mid_price_1_change")
            - pl.col("mid_price_1_change")
        ).alias("recent_mid_price_1_change_delta"),
        (
            pl.col("last60_book_volume_imbalance_1")
            - pl.col("book_volume_imbalance_1")
        ).alias("recent_book_imbalance_1_delta"),
        (pl.col("last60_spread_2_mean") - pl.col("spread_2_mean")).alias(
            "recent_spread_2_delta"
        ),
        (
            pl.col("last60_mid_price_2_change")
            - pl.col("mid_price_2_change")
        ).alias("recent_mid_price_2_change_delta"),
        (
            pl.col("last60_book_volume_imbalance_2")
            - pl.col("book_volume_imbalance_2")
        ).alias("recent_book_imbalance_2_delta"),
    ]

    # 档位关系：直接描述第二档相对第一档的宽度、中心和深度形状。
    # Cross-level features describe level two relative to level one book geometry.
    cross_level_expressions = [
        (pl.col("spread_2_mean") - pl.col("spread_1_mean")).alias(
            "level2_spread_minus_level1_spread"
        ),
        (pl.col("mid_price_2_mean") - pl.col("mid_price_1_mean")).alias(
            "level2_mid_minus_level1_mid"
        ),
        (pl.col("mid_price_2_change") - pl.col("mid_price_1_change")).alias(
            "level2_mid_change_minus_level1_mid_change"
        ),
        (
            pl.col("book_volume_imbalance_2")
            - pl.col("book_volume_imbalance_1")
        ).alias("level2_imbalance_minus_level1_imbalance"),
        safe_ratio(pl.col("ask_volume_2_sum"), pl.col("ask_volume_1_sum")).alias(
            "ask_volume_2_to_1_sum_ratio"
        ),
        safe_ratio(pl.col("bid_volume_2_sum"), pl.col("bid_volume_1_sum")).alias(
            "bid_volume_2_to_1_sum_ratio"
        ),
        safe_ratio(level2_total_volume, level1_total_volume).alias(
            "total_volume_2_to_1_sum_ratio"
        ),
        (
            pl.col("last60_spread_2_mean")
            - pl.col("last60_spread_1_mean")
        ).alias("last60_level2_spread_minus_level1_spread"),
        (
            pl.col("last60_mid_price_2_change")
            - pl.col("last60_mid_price_1_change")
        ).alias("last60_level2_mid_change_minus_level1_mid_change"),
        (
            pl.col("last60_book_volume_imbalance_2")
            - pl.col("last60_book_volume_imbalance_1")
        ).alias("last60_level2_imbalance_minus_level1_imbalance"),
        safe_ratio(recent_level2_total_volume, recent_level1_total_volume).alias(
            "last60_total_volume_2_to_1_mean_ratio"
        ),
    ]

    output = source.select(
        "sample_id", *(temporal_expressions + cross_level_expressions)
    )
    expected_columns = [
        "sample_id", *TEMPORAL_FEATURE_COLUMNS, *CROSS_LEVEL_FEATURE_COLUMNS
    ]
    if output.columns != expected_columns or output.shape != (1_257_637, 23):
        raise AssertionError(
            f"Unexpected relational feature output: {output.shape}, {output.columns}"
        )
    if output["sample_id"].n_unique() != output.height:
        raise AssertionError("Relational feature sample IDs are not unique.")
    infinity_count = output.drop("sample_id").select(
        pl.all().is_infinite().sum()
    ).to_numpy().sum()
    if infinity_count != 0:
        raise AssertionError("Relational features contain infinite values.")
    safe_write_ipc(output, output_path)
    print(f"relational feature shape = {output.shape}")
    print(f"relational feature output = {output_path}")
    print(output.null_count())


if __name__ == "__main__":
    main()
