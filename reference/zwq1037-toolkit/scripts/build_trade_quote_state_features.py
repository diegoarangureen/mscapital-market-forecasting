"""Build target-free trade-at-quote state features from raw event streams.

Each transaction is matched to the latest strictly earlier market snapshot from
the same sample.  The match may be at most six seconds old.  The resulting
features describe trade pressure conditional on the liquidity that was visible
when the trade occurred.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl


WINDOWS = (15, 60)
MATCH_TOLERANCE_SECONDS = 6.0
FEATURE_COLUMNS = [
    "tx_depth_signed_rate_15",
    "tx_depth_gross_rate_15",
    "tx_exec_mid_bps_15",
    "tx_effective_cost_bps_15",
    "tx_book_state_shift_15",
    "tx_thin_book_signed_share_15",
    "tx_depth_signed_rate_60",
    "tx_depth_gross_rate_60",
    "tx_exec_mid_bps_60",
    "tx_effective_cost_bps_60",
    "tx_book_state_shift_60",
    "tx_thin_book_signed_share_60",
    "tx_depth_signed_accel_15_vs_15_60",
    "tx_exec_mid_accel_15_vs_15_60",
    "tx_quote_matched_volume_share_15",
    "tx_quote_matched_volume_share_60",
]


def collect_streaming(frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect a lazy query with Polars' streaming engine.

    使用流式执行降低大文件处理时的峰值内存。
    Use streaming execution to reduce peak memory on large files.
    """

    return frame.collect(engine="streaming")


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a finite ratio and leave undefined rows as null."""

    return (
        pl.when(denominator.is_finite() & (denominator.abs() > 1.0e-12))
        .then(numerator / denominator)
        .otherwise(None)
    )


def load_market(raw_path: Path, max_sample_id: int | None) -> pl.LazyFrame:
    """Project the market columns needed by the as-of match."""

    frame = pl.scan_ipc(raw_path).select(
        "sample_id",
        pl.col("seconds_before_predict").cast(pl.Float64).alias("market_seconds"),
        pl.col("ask_price_1").cast(pl.Float64),
        pl.col("ask_volume_1").cast(pl.Float64),
        pl.col("bid_price_1").cast(pl.Float64),
        pl.col("bid_volume_1").cast(pl.Float64),
    )
    if max_sample_id is not None:
        frame = frame.filter(pl.col("sample_id") < max_sample_id)
    return frame.with_columns(
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("quote_mid"),
        safe_ratio(
            pl.col("bid_volume_1") - pl.col("ask_volume_1"),
            pl.col("bid_volume_1") + pl.col("ask_volume_1"),
        ).alias("quote_imbalance"),
    )


def load_transactions(raw_path: Path, max_sample_id: int | None) -> pl.LazyFrame:
    """Project recent transaction columns and encode trade direction."""

    frame = pl.scan_ipc(raw_path).select(
        "sample_id",
        pl.col("seconds_before_predict").cast(pl.Float64).alias("trade_seconds"),
        pl.col("price").cast(pl.Float64).alias("trade_price"),
        pl.col("volume").cast(pl.Float64).alias("trade_volume"),
        pl.col("side").cast(pl.Int8).alias("trade_side"),
    )
    if max_sample_id is not None:
        frame = frame.filter(pl.col("sample_id") < max_sample_id)
    return frame.filter(pl.col("trade_seconds") <= 60.0).with_columns(
        pl.when(pl.col("trade_side") == 0)
        .then(1.0)
        .when(pl.col("trade_side") == 1)
        .then(-1.0)
        .otherwise(None)
        .alias("trade_sign")
    )


def market_reference(market: pl.LazyFrame, window: int) -> pl.DataFrame:
    """Compute per-sample reference liquidity for one recent window."""

    return collect_streaming(
        market.filter(pl.col("market_seconds") <= float(window))
        .group_by("sample_id")
        .agg(
            pl.col("ask_volume_1").mean().alias(f"ref_ask_depth_{window}"),
            pl.col("bid_volume_1").mean().alias(f"ref_bid_depth_{window}"),
            pl.col("quote_imbalance").mean().alias(f"ref_book_imbalance_{window}"),
        )
        .sort("sample_id")
    )


def match_trades_to_quotes(
    transactions: pl.LazyFrame, market: pl.LazyFrame
) -> pl.DataFrame:
    """Match every trade to the latest strictly earlier quote in its sample."""

    # seconds_before_predict decreases as real time moves forward.  Therefore the
    # strictly earlier quote has a larger value, which is a forward as-of match.
    # seconds_before_predict 随真实时间前进而减小，所以历史盘口使用 forward 匹配。
    left = transactions.sort("sample_id", "trade_seconds")
    right = market.filter(pl.col("market_seconds") <= 66.0).sort(
        "sample_id", "market_seconds"
    )
    matched = left.join_asof(
        right,
        left_on="trade_seconds",
        right_on="market_seconds",
        by="sample_id",
        strategy="forward",
        tolerance=MATCH_TOLERANCE_SECONDS,
        allow_exact_matches=False,
        check_sortedness=False,
    )
    return collect_streaming(
        matched.with_columns(
            pl.when(pl.col("trade_side") == 0)
            .then(pl.col("ask_volume_1"))
            .otherwise(pl.col("bid_volume_1"))
            .alias("opposite_depth")
        ).with_columns(
            (
                pl.col("market_seconds").is_not_null()
                & pl.col("trade_sign").is_not_null()
                & pl.col("opposite_depth").is_finite()
                & (pl.col("opposite_depth") > 0.0)
                & pl.col("trade_volume").is_finite()
                & (pl.col("trade_volume") >= 0.0)
            ).alias("quote_match_valid")
        ).with_columns(
            (
                pl.col("quote_match_valid")
                & pl.col("trade_price").is_finite()
                & (pl.col("trade_price") > 0.0)
                & pl.col("quote_mid").is_finite()
                & (pl.col("quote_mid") > 0.0)
                & (pl.col("ask_price_1") > 0.0)
                & (pl.col("bid_price_1") > 0.0)
                & (pl.col("ask_price_1") >= pl.col("bid_price_1"))
            ).alias("price_match_valid")
        )
    )


def aggregate_window(
    matched: pl.DataFrame, reference: pl.DataFrame, window: int
) -> pl.DataFrame:
    """Aggregate six state features plus match coverage for one window."""

    ref_ask = pl.col(f"ref_ask_depth_{window}")
    ref_bid = pl.col(f"ref_bid_depth_{window}")
    ref_depth = (
        pl.when(pl.col("trade_side") == 0).then(ref_ask).otherwise(ref_bid)
    )
    valid = pl.col("quote_match_valid")
    price_valid = pl.col("price_match_valid")
    volume = pl.col("trade_volume")
    signed_log_depth = pl.col("trade_sign") * (volume / pl.col("opposite_depth")).log1p()
    exec_mid_bps = 1.0e4 * (pl.col("trade_price") - pl.col("quote_mid")) / pl.col("quote_mid")
    thin_weight = ref_depth / (ref_depth + pl.col("opposite_depth"))

    data = matched.filter(pl.col("trade_seconds") <= float(window)).join(
        reference, on="sample_id", how="left"
    )
    sums = data.group_by("sample_id").agg(
        volume.sum().alias("all_volume"),
        pl.when(valid).then(volume).otherwise(0.0).sum().alias("matched_volume"),
        pl.when(price_valid)
        .then(volume)
        .otherwise(0.0)
        .sum()
        .alias("price_matched_volume"),
        pl.when(valid).then(signed_log_depth).otherwise(0.0).sum().alias("signed_depth_sum"),
        pl.when(valid)
        .then((volume / pl.col("opposite_depth")).log1p())
        .otherwise(0.0)
        .sum()
        .alias("gross_depth_sum"),
        pl.when(price_valid)
        .then(volume * exec_mid_bps)
        .otherwise(0.0)
        .sum()
        .alias("exec_mid_num"),
        pl.when(price_valid)
        .then(volume * pl.col("trade_sign") * exec_mid_bps)
        .otherwise(0.0)
        .sum()
        .alias("effective_cost_num"),
        pl.when(valid)
        .then(volume * pl.col("quote_imbalance"))
        .otherwise(0.0)
        .sum()
        .alias("trade_book_num"),
        pl.when(valid)
        .then(volume * pl.col("trade_sign") * thin_weight)
        .otherwise(0.0)
        .sum()
        .alias("thin_signed_num"),
        pl.col(f"ref_book_imbalance_{window}").first(),
    )
    return sums.select(
        "sample_id",
        (pl.col("signed_depth_sum") / float(window)).cast(pl.Float32).alias(
            f"tx_depth_signed_rate_{window}"
        ),
        (pl.col("gross_depth_sum") / float(window)).cast(pl.Float32).alias(
            f"tx_depth_gross_rate_{window}"
        ),
        safe_ratio(pl.col("exec_mid_num"), pl.col("price_matched_volume"))
        .cast(pl.Float32)
        .alias(f"tx_exec_mid_bps_{window}"),
        safe_ratio(pl.col("effective_cost_num"), pl.col("price_matched_volume"))
        .cast(pl.Float32)
        .alias(f"tx_effective_cost_bps_{window}"),
        (
            safe_ratio(pl.col("trade_book_num"), pl.col("matched_volume"))
            - pl.col(f"ref_book_imbalance_{window}")
        )
        .cast(pl.Float32)
        .alias(f"tx_book_state_shift_{window}"),
        safe_ratio(pl.col("thin_signed_num"), pl.col("matched_volume"))
        .cast(pl.Float32)
        .alias(f"tx_thin_book_signed_share_{window}"),
        safe_ratio(pl.col("matched_volume"), pl.col("all_volume"))
        .cast(pl.Float32)
        .alias(f"tx_quote_matched_volume_share_{window}"),
    )


def aggregate_previous_45(matched: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the non-overlapping 15-to-60 second comparison interval."""

    valid = pl.col("quote_match_valid")
    price_valid = pl.col("price_match_valid")
    volume = pl.col("trade_volume")
    signed_log_depth = pl.col("trade_sign") * (volume / pl.col("opposite_depth")).log1p()
    exec_mid_bps = 1.0e4 * (pl.col("trade_price") - pl.col("quote_mid")) / pl.col("quote_mid")
    return (
        matched.filter(
            (pl.col("trade_seconds") > 15.0) & (pl.col("trade_seconds") <= 60.0)
        )
        .group_by("sample_id")
        .agg(
            pl.when(valid).then(signed_log_depth).otherwise(0.0).sum().alias("signed_depth_sum_prev45"),
            pl.when(price_valid)
            .then(volume)
            .otherwise(0.0)
            .sum()
            .alias("price_matched_volume_prev45"),
            pl.when(price_valid)
            .then(volume * exec_mid_bps)
            .otherwise(0.0)
            .sum()
            .alias("exec_mid_num_prev45"),
        )
        .select(
            "sample_id",
            (pl.col("signed_depth_sum_prev45") / 45.0).alias("signed_depth_rate_prev45"),
            safe_ratio(
                pl.col("exec_mid_num_prev45"),
                pl.col("price_matched_volume_prev45"),
            ).alias("exec_mid_bps_prev45"),
        )
    )


def build_features(project_dir: Path, split: str, max_sample_id: int | None) -> pl.DataFrame:
    """Build all sixteen features for one data split."""

    raw_dir = project_dir / "data" / "raw" / split
    market = load_market(raw_dir / "market.feather", max_sample_id)
    transactions = load_transactions(raw_dir / "transaction.feather", max_sample_id)
    sample_ids = collect_streaming(market.select("sample_id").unique().sort("sample_id"))
    references = {window: market_reference(market, window) for window in WINDOWS}
    matched = match_trades_to_quotes(transactions, market)

    output = sample_ids
    aggregates = {}
    for window in WINDOWS:
        aggregates[window] = aggregate_window(matched, references[window], window)
        output = output.join(aggregates[window], on="sample_id", how="left")
    previous = aggregate_previous_45(matched)
    output = output.join(previous, on="sample_id", how="left").with_columns(
        (
            pl.col("tx_depth_signed_rate_15") - pl.col("signed_depth_rate_prev45")
        )
        .cast(pl.Float32)
        .alias("tx_depth_signed_accel_15_vs_15_60"),
        (pl.col("tx_exec_mid_bps_15") - pl.col("exec_mid_bps_prev45"))
        .cast(pl.Float32)
        .alias("tx_exec_mid_accel_15_vs_15_60"),
    )
    return output.select("sample_id", *FEATURE_COLUMNS).sort("sample_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--max-sample-id",
        type=int,
        default=None,
        help="Optional exclusive sample-id bound for a smoke run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    output_path = arguments.output or (
        project_dir
        / "data"
        / "processed"
        / f"{arguments.split}_trade_quote_state_features.feather"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    features = build_features(
        project_dir, arguments.split, arguments.max_sample_id
    )
    if features.width != len(FEATURE_COLUMNS) + 1:
        raise AssertionError("Unexpected feature count.")
    for column in FEATURE_COLUMNS:
        finite = features[column].is_finite() | features[column].is_null()
        if not finite.all():
            raise AssertionError(f"Feature contains infinity: {column}")
    features.write_ipc(output_path, compression="uncompressed")
    summary = {
        "split": arguments.split,
        "rows": features.height,
        "feature_count": len(FEATURE_COLUMNS),
        "max_sample_id": arguments.max_sample_id,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output_path),
        "null_rates": {
            column: features[column].null_count() / max(features.height, 1)
            for column in FEATURE_COLUMNS
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
