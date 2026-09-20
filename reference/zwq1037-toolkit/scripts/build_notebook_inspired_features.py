"""Build deployable multi-horizon flow and exponentially decayed market features.

The formulas are inspired by public competition notebooks, but are reimplemented
against the raw event semantics used by this project.  Train and test always use
the same target-free transformations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from build_event_flow_features import (
    conditional_count,
    conditional_volume,
    master_sample_ids,
    safe_ratio,
    safe_write_ipc,
)
from build_train_market_microstructure_features import (
    BATCH_SIZE,
    CHANNELS,
    normalized_ofi,
    unscale,
)


TRANSACTION_MULTI_FEATURE_COLUMNS = [
    "trade_volume_imbalance_10",
    "trade_volume_imbalance_30",
    "trade_count_imbalance_10",
    "trade_count_imbalance_30",
    "trade_signed_amount_ratio_10",
    "trade_signed_amount_ratio_30",
    "trade_signed_amount_ratio_60",
    "trade_vwap_10_vs_60",
    "trade_vwap_30_vs_60",
    "trade_volume_share_10_of_60",
    "trade_volume_share_30_of_60",
    "trade_count_share_10_of_60",
    "trade_count_share_30_of_60",
    "trade_pressure_delta_10_vs_10_30",
    "trade_pressure_delta_30_vs_30_60",
]

ORDER_MULTI_FEATURE_COLUMNS = [
    "new_order_volume_imbalance_10",
    "new_order_volume_imbalance_30",
    "cancel_order_pressure_10",
    "cancel_order_pressure_30",
    "net_order_pressure_10",
    "net_order_pressure_30",
    "total_cancel_ratio_10",
    "total_cancel_ratio_30",
    "order_volume_share_10_of_60",
    "order_volume_share_30_of_60",
    "order_count_share_10_of_60",
    "order_count_share_30_of_60",
    "net_order_pressure_delta_10_vs_10_30",
    "net_order_pressure_delta_30_vs_30_60",
    "cancel_ratio_delta_10_vs_60",
]

DECAY_SIGNALS = [
    "book_imbalance_1",
    "total_book_imbalance",
    "microprice_displacement",
    "relative_spread",
    "ofi_1",
    "multilevel_ofi",
    "log_total_depth",
]
DECAY_TAUS = [30, 120]
MARKET_DECAY_FEATURE_COLUMNS = [
    *(f"{signal}_ewm_{tau}" for signal in DECAY_SIGNALS for tau in DECAY_TAUS),
    *(f"{signal}_ewm_30_minus_120" for signal in DECAY_SIGNALS),
]


def conditional_amount(condition: pl.Expr, alias: str) -> pl.Expr:
    """Sum price times volume inside one event condition."""

    return (
        pl.when(condition)
        .then(pl.col("price").cast(pl.Float64) * pl.col("volume").cast(pl.Float64))
        .otherwise(0.0)
        .sum()
        .alias(alias)
    )


def build_transaction_multi(project_dir: Path, split: str) -> Path:
    """Build nested-window transaction pressure, activity, and VWAP features."""

    seconds = pl.col("seconds_before_predict")
    side = pl.col("side")
    windows = {
        "10": seconds <= 10.0,
        "30": seconds <= 30.0,
        "60": seconds <= 60.0,
        "10_30": (seconds > 10.0) & (seconds <= 30.0),
        "30_60": (seconds > 30.0) & (seconds <= 60.0),
    }
    expressions: list[pl.Expr] = []
    for name, window in windows.items():
        buy = window & (side == 0)
        sell = window & (side == 1)
        expressions.extend(
            [
                conditional_volume(buy, f"buy_volume_{name}"),
                conditional_volume(sell, f"sell_volume_{name}"),
                conditional_count(buy, f"buy_count_{name}"),
                conditional_count(sell, f"sell_count_{name}"),
                conditional_amount(buy, f"buy_amount_{name}"),
                conditional_amount(sell, f"sell_amount_{name}"),
            ]
        )

    input_path = project_dir / "data" / "raw" / split / "transaction.feather"
    aggregates = (
        pl.scan_ipc(input_path)
        .select(["sample_id", "seconds_before_predict", "price", "volume", "side"])
        .group_by("sample_id")
        .agg(expressions)
        .collect(engine="streaming")
    )
    aggregates = master_sample_ids(project_dir, split).join(
        aggregates, on="sample_id", how="left", validate="1:1"
    )
    value_columns = [column for column in aggregates.columns if column != "sample_id"]
    aggregates = aggregates.with_columns(pl.col(value_columns).fill_null(0))

    def volume_imbalance(name: str) -> pl.Expr:
        buy = pl.col(f"buy_volume_{name}")
        sell = pl.col(f"sell_volume_{name}")
        return safe_ratio(buy - sell, buy + sell)

    def count_imbalance(name: str) -> pl.Expr:
        buy = pl.col(f"buy_count_{name}")
        sell = pl.col(f"sell_count_{name}")
        return safe_ratio(buy - sell, buy + sell)

    def signed_amount_ratio(name: str) -> pl.Expr:
        buy = pl.col(f"buy_amount_{name}")
        sell = pl.col(f"sell_amount_{name}")
        return safe_ratio(buy - sell, buy + sell)

    def total_volume(name: str) -> pl.Expr:
        return pl.col(f"buy_volume_{name}") + pl.col(f"sell_volume_{name}")

    def total_count(name: str) -> pl.Expr:
        return pl.col(f"buy_count_{name}") + pl.col(f"sell_count_{name}")

    def vwap(name: str) -> pl.Expr:
        return safe_ratio(
            pl.col(f"buy_amount_{name}") + pl.col(f"sell_amount_{name}"),
            total_volume(name),
        )

    features = aggregates.with_columns(
        volume_imbalance("10").alias("trade_volume_imbalance_10"),
        volume_imbalance("30").alias("trade_volume_imbalance_30"),
        count_imbalance("10").alias("trade_count_imbalance_10"),
        count_imbalance("30").alias("trade_count_imbalance_30"),
        signed_amount_ratio("10").alias("trade_signed_amount_ratio_10"),
        signed_amount_ratio("30").alias("trade_signed_amount_ratio_30"),
        signed_amount_ratio("60").alias("trade_signed_amount_ratio_60"),
        safe_ratio(vwap("10") - vwap("60"), vwap("60")).alias("trade_vwap_10_vs_60"),
        safe_ratio(vwap("30") - vwap("60"), vwap("60")).alias("trade_vwap_30_vs_60"),
        safe_ratio(total_volume("10"), total_volume("60")).alias("trade_volume_share_10_of_60"),
        safe_ratio(total_volume("30"), total_volume("60")).alias("trade_volume_share_30_of_60"),
        safe_ratio(total_count("10"), total_count("60")).alias("trade_count_share_10_of_60"),
        safe_ratio(total_count("30"), total_count("60")).alias("trade_count_share_30_of_60"),
        (volume_imbalance("10") - volume_imbalance("10_30")).alias(
            "trade_pressure_delta_10_vs_10_30"
        ),
        (volume_imbalance("30") - volume_imbalance("30_60")).alias(
            "trade_pressure_delta_30_vs_30_60"
        ),
    ).select("sample_id", *TRANSACTION_MULTI_FEATURE_COLUMNS)
    output_path = project_dir / "data" / "processed" / f"{split}_transaction_multiwindow_features.feather"
    safe_write_ipc(features, output_path)
    print(f"transaction multi-window shape = {features.shape}", flush=True)
    print(f"transaction multi-window output = {output_path}", flush=True)
    return output_path


def build_order_multi(project_dir: Path, split: str) -> Path:
    """Build nested-window new/cancel pressure and activity features."""

    seconds = pl.col("seconds_before_predict")
    side = pl.col("side")
    action = pl.col("order_action")
    windows = {
        "10": seconds <= 10.0,
        "30": seconds <= 30.0,
        "60": seconds <= 60.0,
        "10_30": (seconds > 10.0) & (seconds <= 30.0),
        "30_60": (seconds > 30.0) & (seconds <= 60.0),
    }
    expressions: list[pl.Expr] = []
    for name, window in windows.items():
        for side_name, side_code in [("buy", 0), ("sell", 1)]:
            for action_name, action_code in [("new", 0), ("cancel", 1)]:
                condition = window & (side == side_code) & (action == action_code)
                expressions.append(
                    conditional_volume(condition, f"{action_name}_{side_name}_volume_{name}")
                )
        expressions.append(conditional_count(window, f"order_count_{name}"))

    input_path = project_dir / "data" / "raw" / split / "order.feather"
    aggregates = (
        pl.scan_ipc(input_path)
        .select(["sample_id", "seconds_before_predict", "volume", "side", "order_action"])
        .group_by("sample_id")
        .agg(expressions)
        .collect(engine="streaming")
    )
    aggregates = master_sample_ids(project_dir, split).join(
        aggregates, on="sample_id", how="left", validate="1:1"
    )
    value_columns = [column for column in aggregates.columns if column != "sample_id"]
    aggregates = aggregates.with_columns(pl.col(value_columns).fill_null(0))

    def components(name: str) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        return (
            pl.col(f"new_buy_volume_{name}"),
            pl.col(f"new_sell_volume_{name}"),
            pl.col(f"cancel_buy_volume_{name}"),
            pl.col(f"cancel_sell_volume_{name}"),
        )

    def total_volume(name: str) -> pl.Expr:
        return sum(components(name), start=pl.lit(0))

    def new_imbalance(name: str) -> pl.Expr:
        new_buy, new_sell, _, _ = components(name)
        return safe_ratio(new_buy - new_sell, new_buy + new_sell)

    def cancel_pressure(name: str) -> pl.Expr:
        _, _, cancel_buy, cancel_sell = components(name)
        return safe_ratio(cancel_sell - cancel_buy, cancel_buy + cancel_sell)

    def net_pressure(name: str) -> pl.Expr:
        new_buy, new_sell, cancel_buy, cancel_sell = components(name)
        return safe_ratio(
            new_buy - new_sell - cancel_buy + cancel_sell,
            new_buy + new_sell + cancel_buy + cancel_sell,
        )

    def cancel_ratio(name: str) -> pl.Expr:
        _, _, cancel_buy, cancel_sell = components(name)
        return safe_ratio(cancel_buy + cancel_sell, total_volume(name))

    features = aggregates.with_columns(
        new_imbalance("10").alias("new_order_volume_imbalance_10"),
        new_imbalance("30").alias("new_order_volume_imbalance_30"),
        cancel_pressure("10").alias("cancel_order_pressure_10"),
        cancel_pressure("30").alias("cancel_order_pressure_30"),
        net_pressure("10").alias("net_order_pressure_10"),
        net_pressure("30").alias("net_order_pressure_30"),
        cancel_ratio("10").alias("total_cancel_ratio_10"),
        cancel_ratio("30").alias("total_cancel_ratio_30"),
        safe_ratio(total_volume("10"), total_volume("60")).alias("order_volume_share_10_of_60"),
        safe_ratio(total_volume("30"), total_volume("60")).alias("order_volume_share_30_of_60"),
        safe_ratio(pl.col("order_count_10"), pl.col("order_count_60")).alias("order_count_share_10_of_60"),
        safe_ratio(pl.col("order_count_30"), pl.col("order_count_60")).alias("order_count_share_30_of_60"),
        (net_pressure("10") - net_pressure("10_30")).alias(
            "net_order_pressure_delta_10_vs_10_30"
        ),
        (net_pressure("30") - net_pressure("30_60")).alias(
            "net_order_pressure_delta_30_vs_30_60"
        ),
        (cancel_ratio("10") - cancel_ratio("60")).alias("cancel_ratio_delta_10_vs_60"),
    ).select("sample_id", *ORDER_MULTI_FEATURE_COLUMNS)
    output_path = project_dir / "data" / "processed" / f"{split}_order_multiwindow_features.feather"
    safe_write_ipc(features, output_path)
    print(f"order multi-window shape = {features.shape}", flush=True)
    print(f"order multi-window output = {output_path}", flush=True)
    return output_path


def masked_weighted_mean(
    values: np.ndarray,
    mask: np.ndarray,
    seconds: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Return a normalized exponentially decayed mean for every sample."""

    weights = np.exp(-np.maximum(seconds, 0.0) / tau)
    valid = mask & np.isfinite(values) & np.isfinite(seconds)
    weighted_values = np.where(valid, values * weights, 0.0)
    weight_sum = np.where(valid, weights, 0.0).sum(axis=1, dtype=np.float64)
    value_sum = weighted_values.sum(axis=1, dtype=np.float64)
    return np.divide(
        value_sum,
        weight_sum,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=weight_sum > 0,
    )


def build_market_decay_batch(cache_block: np.ndarray, statistics: dict) -> dict[str, np.ndarray]:
    """Build robust decayed book signals from one bounded sequence-cache batch."""

    row_mask = cache_block[:, CHANNELS["row_mask"], :] > 0.5
    seconds = cache_block[:, CHANNELS["seconds_before_predict"], :].astype(np.float64) * 600.0
    ask_1 = unscale(cache_block, "ask_price_1", statistics)
    bid_1 = unscale(cache_block, "bid_price_1", statistics)
    ask_volume_1 = unscale(cache_block, "ask_volume_1", statistics)
    bid_volume_1 = unscale(cache_block, "bid_volume_1", statistics)
    ask_2 = unscale(cache_block, "ask_price_2", statistics)
    bid_2 = unscale(cache_block, "bid_price_2", statistics)
    ask_volume_2 = unscale(cache_block, "ask_volume_2", statistics)
    bid_volume_2 = unscale(cache_block, "bid_volume_2", statistics)

    mid = (ask_1 + bid_1) / 2.0
    relative_spread = np.divide(
        ask_1 - bid_1,
        mid,
        out=np.full_like(mid, np.nan),
        where=mid != 0,
    )
    valid_mid = (
        row_mask
        & np.isfinite(mid)
        & (mid >= 0.1)
        & (relative_spread >= 0.0)
        & (relative_spread <= 0.1)
    )
    valid_depth_1 = row_mask & (ask_volume_1 > 0) & (bid_volume_1 > 0)
    valid_depth_2 = row_mask & (ask_volume_2 > 0) & (bid_volume_2 > 0)
    depth_1 = ask_volume_1 + bid_volume_1
    depth_2 = ask_volume_2 + bid_volume_2
    total_depth = depth_1 + depth_2
    valid_total_depth = valid_depth_1 & valid_depth_2
    imbalance_1 = np.divide(
        bid_volume_1 - ask_volume_1,
        depth_1,
        out=np.full_like(depth_1, np.nan),
        where=valid_depth_1,
    )
    total_imbalance = np.divide(
        bid_volume_1 + bid_volume_2 - ask_volume_1 - ask_volume_2,
        total_depth,
        out=np.full_like(total_depth, np.nan),
        where=valid_total_depth,
    )
    microprice = np.divide(
        ask_1 * bid_volume_1 + bid_1 * ask_volume_1,
        depth_1,
        out=np.full_like(mid, np.nan),
        where=valid_depth_1 & valid_mid,
    )
    microprice_displacement = np.divide(
        microprice - mid,
        mid,
        out=np.full_like(mid, np.nan),
        where=valid_depth_1 & valid_mid,
    )
    log_total_depth = np.log1p(np.where(valid_total_depth, total_depth, np.nan))
    ofi_1, ofi_mask_1 = normalized_ofi(
        bid_1, bid_volume_1, ask_1, ask_volume_1, row_mask
    )
    ofi_2, ofi_mask_2 = normalized_ofi(
        bid_2, bid_volume_2, ask_2, ask_volume_2, row_mask
    )
    multilevel_mask = ofi_mask_1 & ofi_mask_2
    multilevel_ofi = ofi_1 + ofi_2
    event_seconds = seconds[:, 1:]

    snapshot_signals = {
        "book_imbalance_1": (imbalance_1, valid_depth_1, seconds),
        "total_book_imbalance": (total_imbalance, valid_total_depth, seconds),
        "microprice_displacement": (
            microprice_displacement,
            valid_depth_1 & valid_mid,
            seconds,
        ),
        "relative_spread": (relative_spread, valid_mid, seconds),
        "log_total_depth": (log_total_depth, valid_total_depth, seconds),
        "ofi_1": (ofi_1, ofi_mask_1, event_seconds),
        "multilevel_ofi": (multilevel_ofi, multilevel_mask, event_seconds),
    }
    features: dict[str, np.ndarray] = {}
    for signal in DECAY_SIGNALS:
        values, mask, signal_seconds = snapshot_signals[signal]
        for tau in DECAY_TAUS:
            features[f"{signal}_ewm_{tau}"] = masked_weighted_mean(
                values, mask, signal_seconds, float(tau)
            )
        features[f"{signal}_ewm_30_minus_120"] = (
            features[f"{signal}_ewm_30"] - features[f"{signal}_ewm_120"]
        )
    return features


def build_market_decay(project_dir: Path, split: str) -> Path:
    """Build exponentially decayed market signals from the existing sequence cache."""

    cache_dir = project_dir / "data" / "processed" / f"{split}_sequence_cache_v1"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    statistics = {}
    for name in CHANNELS:
        statistics_path = cache_dir / "statistics" / f"{name}.json"
        if statistics_path.exists():
            statistics[name] = json.loads(statistics_path.read_text(encoding="utf-8"))

    output = np.empty((cache.shape[0], len(MARKET_DECAY_FEATURE_COLUMNS)), dtype=np.float32)
    for start in range(0, cache.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, cache.shape[0])
        batch_features = build_market_decay_batch(cache[start:end], statistics)
        if list(batch_features) != MARKET_DECAY_FEATURE_COLUMNS:
            raise AssertionError("Market-decay feature order does not match the declared schema.")
        for index, name in enumerate(MARKET_DECAY_FEATURE_COLUMNS):
            output[start:end, index] = batch_features[name]
        if start % 100_000 == 0:
            print(f"market decay processed samples {start}:{end}", flush=True)
    if np.isinf(output).any():
        raise ValueError("Market-decay features contain infinity.")

    ids = master_sample_ids(project_dir, split).sort("sample_id")["sample_id"].to_numpy()
    if len(ids) != len(output):
        raise AssertionError("Sequence cache and master sample IDs have different row counts.")
    features = pd.DataFrame(output, columns=MARKET_DECAY_FEATURE_COLUMNS)
    features.insert(0, "sample_id", ids)
    output_path = project_dir / "data" / "processed" / f"{split}_market_decay_features.feather"
    safe_write_ipc(pl.from_pandas(features), output_path)
    print(f"market decay shape = {features.shape}", flush=True)
    print(f"market decay output = {output_path}", flush=True)
    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--source",
        choices=["transaction", "order", "market", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parents[1]
    if arguments.source in {"transaction", "all"}:
        build_transaction_multi(project_dir, arguments.split)
    if arguments.source in {"order", "all"}:
        build_order_multi(project_dir, arguments.split)
    if arguments.source in {"market", "all"}:
        build_market_decay(project_dir, arguments.split)


if __name__ == "__main__":
    main()
