"""Build target-free order-position features from order and market streams."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl


MATCH_TOLERANCE_SECONDS = 6.0
WINDOWS = (15, 60)
FEATURE_COLUMNS = [
    "order_crossable_new_direction_share_15",
    "order_inside_new_direction_share_15",
    "order_near_new_pressure_15",
    "order_near_cancel_pressure_15",
    "order_near_net_pressure_15",
    "order_cancel_ratio_side_diff_15",
    "order_new_closeness_side_diff_15",
    "order_depth_adjusted_net_rate_15",
    "order_crossable_new_direction_share_60",
    "order_inside_new_direction_share_60",
    "order_near_new_pressure_60",
    "order_near_cancel_pressure_60",
    "order_near_net_pressure_60",
    "order_cancel_ratio_side_diff_60",
    "order_new_closeness_side_diff_60",
    "order_depth_adjusted_net_rate_60",
    "order_near_new_accel_15_vs_15_60",
    "order_near_cancel_accel_15_vs_15_60",
    "order_quote_matched_volume_share_15",
    "order_quote_matched_volume_share_60",
]


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a ratio only where its denominator is finite and nonzero."""

    return (
        pl.when(denominator.is_finite() & (denominator.abs() > 1.0e-12))
        .then(numerator / denominator)
        .otherwise(None)
    )


def conditional_sum(mask: pl.Expr, value: pl.Expr, name: str) -> pl.Expr:
    """Sum a row expression under a Boolean mask."""

    return pl.when(mask).then(value).otherwise(0.0).sum().alias(name)


def load_market(path: Path, max_sample_id: int | None) -> pl.LazyFrame:
    """Read only the two-level quote fields needed for order positioning."""

    frame = pl.scan_ipc(path).select(
        "sample_id",
        pl.col("seconds_before_predict").cast(pl.Float64).alias("market_seconds"),
        pl.col("ask_price_1").cast(pl.Float64),
        pl.col("ask_volume_1").cast(pl.Float64),
        pl.col("bid_price_1").cast(pl.Float64),
        pl.col("bid_volume_1").cast(pl.Float64),
        pl.col("ask_price_2").cast(pl.Float64),
        pl.col("bid_price_2").cast(pl.Float64),
    )
    if max_sample_id is not None:
        frame = frame.filter(pl.col("sample_id") < max_sample_id)
    return frame.filter(pl.col("market_seconds") <= 66.0)


def load_orders(path: Path, max_sample_id: int | None) -> pl.LazyFrame:
    """Read the order events and encode side/action signs."""

    frame = pl.scan_ipc(path).select(
        "sample_id",
        pl.col("seconds_before_predict").cast(pl.Float64).alias("order_seconds"),
        pl.col("price").cast(pl.Float64).alias("order_price"),
        pl.col("volume").cast(pl.Float64).alias("order_volume"),
        pl.col("side").cast(pl.Int8).alias("order_side"),
        pl.col("order_action").cast(pl.Int8),
    )
    if max_sample_id is not None:
        frame = frame.filter(pl.col("sample_id") < max_sample_id)
    return frame.filter(pl.col("order_seconds") <= 60.0).with_columns(
        pl.when(pl.col("order_side") == 0)
        .then(1.0)
        .when(pl.col("order_side") == 1)
        .then(-1.0)
        .otherwise(None)
        .alias("side_sign"),
        pl.when(pl.col("order_action") == 0)
        .then(1.0)
        .when(pl.col("order_action") == 1)
        .then(-1.0)
        .otherwise(None)
        .alias("action_sign"),
    )


def matched_orders(orders: pl.LazyFrame, market: pl.LazyFrame) -> pl.LazyFrame:
    """Attach the latest strictly earlier quote to every order event."""

    joined = orders.sort("sample_id", "order_seconds").join_asof(
        market.sort("sample_id", "market_seconds"),
        left_on="order_seconds",
        right_on="market_seconds",
        by="sample_id",
        strategy="forward",
        tolerance=MATCH_TOLERANCE_SECONDS,
        allow_exact_matches=False,
        check_sortedness=False,
    )
    return (
        joined.with_columns(
            pl.when(pl.col("order_side") == 0)
            .then(pl.col("bid_price_1"))
            .otherwise(pl.col("ask_price_1"))
            .alias("own_best_price"),
            pl.when(pl.col("order_side") == 0)
            .then(pl.col("ask_price_1"))
            .otherwise(pl.col("bid_price_1"))
            .alias("opposite_best_price"),
            pl.when(pl.col("order_side") == 0)
            .then(pl.col("bid_volume_1"))
            .otherwise(pl.col("ask_volume_1"))
            .alias("own_depth"),
            pl.when(pl.col("order_side") == 0)
            .then(pl.col("bid_price_1") - pl.col("bid_price_2"))
            .otherwise(pl.col("ask_price_2") - pl.col("ask_price_1"))
            .alias("own_tick"),
            (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        )
        .with_columns(
            pl.max_horizontal("spread", "own_tick", pl.lit(1.0e-8)).alias("price_scale")
        )
        .with_columns(
            (
                pl.col("market_seconds").is_not_null()
                & pl.col("side_sign").is_not_null()
                & pl.col("action_sign").is_not_null()
                & pl.col("order_price").is_finite()
                & (pl.col("order_price") > 0.0)
                & (pl.col("ask_price_1") > 0.0)
                & (pl.col("bid_price_1") > 0.0)
                & (pl.col("ask_price_1") >= pl.col("bid_price_1"))
                & pl.col("own_best_price").is_finite()
                & pl.col("opposite_best_price").is_finite()
                & pl.col("own_depth").is_finite()
                & (pl.col("own_depth") > 0.0)
                & pl.col("order_volume").is_finite()
                & (pl.col("order_volume") >= 0.0)
            ).alias("match_valid")
        )
        .with_columns(
            (
                -(pl.col("order_price") - pl.col("own_best_price")).abs()
                / pl.col("price_scale")
            )
            .clip(-30.0, 0.0)
            .exp()
            .alias("near_weight"),
            (
                ((pl.col("order_side") == 0) & (pl.col("order_price") >= pl.col("ask_price_1")))
                | ((pl.col("order_side") == 1) & (pl.col("order_price") <= pl.col("bid_price_1")))
            ).alias("is_crossable"),
            (
                (pl.col("order_price") > pl.col("bid_price_1"))
                & (pl.col("order_price") < pl.col("ask_price_1"))
            ).alias("is_inside_spread"),
        )
    )


def window_aggregations(window: int, mask: pl.Expr) -> list[pl.Expr]:
    """Declare sufficient statistics for the eight features in one window."""

    valid = mask & pl.col("match_valid")
    new = valid & (pl.col("order_action") == 0)
    cancel = valid & (pl.col("order_action") == 1)
    buy = valid & (pl.col("order_side") == 0)
    sell = valid & (pl.col("order_side") == 1)
    volume = pl.col("order_volume")
    near_volume = volume * pl.col("near_weight")
    suffix = str(window)
    return [
        conditional_sum(mask, volume, f"all_volume_{suffix}"),
        conditional_sum(valid, volume, f"matched_volume_{suffix}"),
        conditional_sum(new, volume, f"new_volume_{suffix}"),
        conditional_sum(cancel, volume, f"cancel_volume_{suffix}"),
        conditional_sum(valid, volume, f"valid_volume_{suffix}"),
        conditional_sum(
            new & pl.col("is_crossable"),
            pl.col("side_sign") * volume,
            f"crossable_new_signed_{suffix}",
        ),
        conditional_sum(
            new & pl.col("is_inside_spread"),
            pl.col("side_sign") * volume,
            f"inside_new_signed_{suffix}",
        ),
        conditional_sum(new, pl.col("side_sign") * near_volume, f"near_new_signed_{suffix}"),
        conditional_sum(cancel, -pl.col("side_sign") * near_volume, f"near_cancel_signed_{suffix}"),
        conditional_sum(
            valid,
            pl.col("action_sign") * pl.col("side_sign") * near_volume,
            f"near_net_signed_{suffix}",
        ),
        conditional_sum(buy & (pl.col("order_action") == 1), near_volume, f"buy_cancel_near_{suffix}"),
        conditional_sum(buy & (pl.col("order_action") == 0), near_volume, f"buy_new_near_{suffix}"),
        conditional_sum(sell & (pl.col("order_action") == 1), near_volume, f"sell_cancel_near_{suffix}"),
        conditional_sum(sell & (pl.col("order_action") == 0), near_volume, f"sell_new_near_{suffix}"),
        conditional_sum(new & (pl.col("order_side") == 0), near_volume, f"buy_new_close_num_{suffix}"),
        conditional_sum(new & (pl.col("order_side") == 0), volume, f"buy_new_close_den_{suffix}"),
        conditional_sum(new & (pl.col("order_side") == 1), near_volume, f"sell_new_close_num_{suffix}"),
        conditional_sum(new & (pl.col("order_side") == 1), volume, f"sell_new_close_den_{suffix}"),
        conditional_sum(
            valid,
            pl.col("action_sign")
            * pl.col("side_sign")
            * pl.col("near_weight")
            * (volume / pl.col("own_depth")).log1p(),
            f"depth_net_sum_{suffix}",
        ),
    ]


def derive_window_features(window: int) -> list[pl.Expr]:
    """Convert sufficient statistics into the eight normalized features."""

    suffix = str(window)
    buy_cancel_ratio = safe_ratio(
        pl.col(f"buy_cancel_near_{suffix}"),
        pl.col(f"buy_cancel_near_{suffix}") + pl.col(f"buy_new_near_{suffix}"),
    )
    sell_cancel_ratio = safe_ratio(
        pl.col(f"sell_cancel_near_{suffix}"),
        pl.col(f"sell_cancel_near_{suffix}") + pl.col(f"sell_new_near_{suffix}"),
    )
    buy_closeness = safe_ratio(
        pl.col(f"buy_new_close_num_{suffix}"),
        pl.col(f"buy_new_close_den_{suffix}"),
    )
    sell_closeness = safe_ratio(
        pl.col(f"sell_new_close_num_{suffix}"),
        pl.col(f"sell_new_close_den_{suffix}"),
    )
    return [
        safe_ratio(pl.col(f"crossable_new_signed_{suffix}"), pl.col(f"new_volume_{suffix}"))
        .cast(pl.Float32)
        .alias(f"order_crossable_new_direction_share_{suffix}"),
        safe_ratio(pl.col(f"inside_new_signed_{suffix}"), pl.col(f"new_volume_{suffix}"))
        .cast(pl.Float32)
        .alias(f"order_inside_new_direction_share_{suffix}"),
        safe_ratio(pl.col(f"near_new_signed_{suffix}"), pl.col(f"new_volume_{suffix}"))
        .cast(pl.Float32)
        .alias(f"order_near_new_pressure_{suffix}"),
        safe_ratio(pl.col(f"near_cancel_signed_{suffix}"), pl.col(f"cancel_volume_{suffix}"))
        .cast(pl.Float32)
        .alias(f"order_near_cancel_pressure_{suffix}"),
        safe_ratio(pl.col(f"near_net_signed_{suffix}"), pl.col(f"valid_volume_{suffix}"))
        .cast(pl.Float32)
        .alias(f"order_near_net_pressure_{suffix}"),
        (sell_cancel_ratio - buy_cancel_ratio)
        .cast(pl.Float32)
        .alias(f"order_cancel_ratio_side_diff_{suffix}"),
        (buy_closeness - sell_closeness)
        .cast(pl.Float32)
        .alias(f"order_new_closeness_side_diff_{suffix}"),
        (pl.col(f"depth_net_sum_{suffix}") / float(window))
        .cast(pl.Float32)
        .alias(f"order_depth_adjusted_net_rate_{suffix}"),
        safe_ratio(pl.col(f"matched_volume_{suffix}"), pl.col(f"all_volume_{suffix}"))
        .cast(pl.Float32)
        .alias(f"order_quote_matched_volume_share_{suffix}"),
    ]


def build_features(project_dir: Path, split: str, max_sample_id: int | None) -> pl.DataFrame:
    """Build all twenty features in one lazy aggregation pass."""

    raw_dir = project_dir / "data" / "raw" / split
    market = load_market(raw_dir / "market.feather", max_sample_id)
    orders = load_orders(raw_dir / "order.feather", max_sample_id)
    joined = matched_orders(orders, market)
    aggregations: list[pl.Expr] = []
    for window in WINDOWS:
        aggregations.extend(
            window_aggregations(window, pl.col("order_seconds") <= float(window))
        )
    aggregations.extend(
        window_aggregations(
            45,
            (pl.col("order_seconds") > 15.0) & (pl.col("order_seconds") <= 60.0),
        )
    )
    grouped = joined.group_by("sample_id").agg(*aggregations)
    features = grouped.with_columns(
        *derive_window_features(15),
        *derive_window_features(60),
        *derive_window_features(45),
    ).with_columns(
        (
            pl.col("order_near_new_pressure_15")
            - pl.col("order_near_new_pressure_45")
        )
        .cast(pl.Float32)
        .alias("order_near_new_accel_15_vs_15_60"),
        (
            pl.col("order_near_cancel_pressure_15")
            - pl.col("order_near_cancel_pressure_45")
        )
        .cast(pl.Float32)
        .alias("order_near_cancel_accel_15_vs_15_60"),
    )
    return features.select("sample_id", *FEATURE_COLUMNS).sort("sample_id").collect(
        engine="streaming"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--max-sample-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    output_path = arguments.output or (
        project_dir
        / "data"
        / "processed"
        / f"{arguments.split}_order_quote_position_features.feather"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    features = build_features(project_dir, arguments.split, arguments.max_sample_id)
    if features.width != len(FEATURE_COLUMNS) + 1:
        raise AssertionError("Unexpected feature count.")
    for column in FEATURE_COLUMNS:
        if not (features[column].is_finite() | features[column].is_null()).all():
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
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
