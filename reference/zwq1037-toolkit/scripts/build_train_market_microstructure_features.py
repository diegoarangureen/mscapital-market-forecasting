"""Build robust price-dynamics and order-book microstructure features.

The existing sequence cache keeps the latest 200 market snapshots per sample.
This script reads it in bounded batches and never loads the 4.4-GiB raw market
table.  All features are target-free and therefore safe to build for train/test.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


BATCH_SIZE = 10_000
CHANNELS = {
    "transaction_avgprice": 0,
    "transaction_volume": 1,
    "transaction_count": 2,
    "ask_price_1": 3,
    "ask_volume_1": 4,
    "bid_price_1": 5,
    "bid_volume_1": 6,
    "ask_price_2": 7,
    "ask_volume_2": 8,
    "bid_price_2": 9,
    "bid_volume_2": 10,
    "seconds_before_predict": 11,
    "has_transactions": 12,
    "row_mask": 13,
}

PRICE_FEATURE_COLUMNS = [
    "mid_return_full",
    "mid_return_180",
    "mid_return_60",
    "mid_return_20",
    "realized_volatility_full",
    "realized_volatility_180",
    "realized_volatility_60",
    "realized_volatility_20",
    "mid_return_std_full",
    "mid_return_skew_full",
    "mid_return_kurtosis_full",
    "mid_momentum_full",
    "mid_momentum_60",
    "mid_last_vs_full_mean",
    "mid_last_vs_60_mean",
    "relative_spread_mean",
    "relative_spread_std",
    "relative_spread_last",
    "relative_spread_60_mean",
    "trade_price_vs_mid_mean",
    "trade_price_vs_mid_last",
]

BOOK_FEATURE_COLUMNS = [
    "book_imbalance_1_mean_robust",
    "book_imbalance_1_std_robust",
    "book_imbalance_1_last_robust",
    "book_imbalance_1_60_mean_robust",
    "book_imbalance_1_20_mean_robust",
    "book_imbalance_2_mean_robust",
    "book_imbalance_2_std_robust",
    "book_imbalance_2_last_robust",
    "book_imbalance_2_60_mean_robust",
    "total_book_imbalance_mean",
    "total_book_imbalance_last",
    "total_book_imbalance_60_mean",
    "microprice_displacement_mean",
    "microprice_displacement_std",
    "microprice_displacement_last",
    "microprice_displacement_60_mean",
    "microprice_displacement_20_mean",
    "ofi_1_mean",
    "ofi_1_std",
    "ofi_1_sum",
    "ofi_1_last",
    "ofi_1_60_mean",
    "ofi_1_20_mean",
    "ofi_1_acceleration",
    "ofi_2_mean",
    "ofi_2_sum",
    "ofi_2_60_mean",
    "multilevel_ofi_mean",
    "multilevel_ofi_sum",
    "multilevel_ofi_60_mean",
    "multilevel_ofi_acceleration",
    "log_total_depth_mean",
    "log_total_depth_std",
    "log_total_depth_last",
    "log_total_depth_60_mean",
]


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return one mean per row while ignoring invalid positions."""

    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1, dtype=np.float64)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def masked_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a sum and keep all-missing rows as NaN instead of zero."""

    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1, dtype=np.float64)
    total[count == 0] = np.nan
    return total


def masked_std(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the population standard deviation for each row."""

    mean = masked_mean(values, mask)
    centered = np.where(mask, values - mean[:, None], 0.0)
    count = mask.sum(axis=1)
    variance = np.divide(
        np.square(centered).sum(axis=1, dtype=np.float64),
        count,
        out=np.full(len(values), np.nan),
        where=count > 0,
    )
    return np.sqrt(variance)


def masked_last(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the final valid value in every right-aligned sequence."""

    reverse_index = np.argmax(mask[:, ::-1], axis=1)
    index = values.shape[1] - 1 - reverse_index
    result = values[np.arange(len(values)), index].astype(np.float64, copy=True)
    result[~mask.any(axis=1)] = np.nan
    return result


def endpoint_return(price: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return (last - first) / first inside one selected window."""

    first_index = np.argmax(mask, axis=1)
    reverse_index = np.argmax(mask[:, ::-1], axis=1)
    last_index = price.shape[1] - 1 - reverse_index
    rows = np.arange(len(price))
    first = price[rows, first_index]
    last = price[rows, last_index]
    enough = mask.sum(axis=1) >= 2
    valid = enough & np.isfinite(first) & np.isfinite(last) & (first > 0)
    return np.divide(last - first, first, out=np.full(len(price), np.nan), where=valid)


def masked_slope(y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the least-squares slope of y against forward-moving time x."""

    x_mean = masked_mean(x, mask)
    y_mean = masked_mean(y, mask)
    x_centered = np.where(mask, x - x_mean[:, None], 0.0)
    y_centered = np.where(mask, y - y_mean[:, None], 0.0)
    numerator = (x_centered * y_centered).sum(axis=1, dtype=np.float64)
    denominator = np.square(x_centered).sum(axis=1, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(y), np.nan),
        where=denominator > 0,
    )


def return_moments(log_returns: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return standard deviation, skewness, and excess kurtosis by sample."""

    mean = masked_mean(log_returns, mask)
    centered = np.where(mask, log_returns - mean[:, None], 0.0)
    count = mask.sum(axis=1)
    second = np.divide(
        np.square(centered).sum(axis=1, dtype=np.float64),
        count,
        out=np.full(len(log_returns), np.nan),
        where=count > 0,
    )
    third = np.divide(
        np.power(centered, 3).sum(axis=1, dtype=np.float64),
        count,
        out=np.full(len(log_returns), np.nan),
        where=count > 0,
    )
    fourth = np.divide(
        np.power(centered, 4).sum(axis=1, dtype=np.float64),
        count,
        out=np.full(len(log_returns), np.nan),
        where=count > 0,
    )
    skew = np.divide(
        third,
        np.power(second, 1.5),
        out=np.full(len(log_returns), np.nan),
        where=second > 0,
    )
    kurtosis = np.divide(
        fourth,
        np.square(second),
        out=np.full(len(log_returns), np.nan),
        where=second > 0,
    ) - 3.0
    return np.sqrt(second), skew, kurtosis


def normalized_ofi(
    bid_price: np.ndarray,
    bid_volume: np.ndarray,
    ask_price: np.ndarray,
    ask_volume: np.ndarray,
    row_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute depth-normalized snapshot OFI using the Cont-style rule."""

    previous_bid_price = bid_price[:, :-1]
    current_bid_price = bid_price[:, 1:]
    previous_ask_price = ask_price[:, :-1]
    current_ask_price = ask_price[:, 1:]
    previous_bid_volume = bid_volume[:, :-1]
    current_bid_volume = bid_volume[:, 1:]
    previous_ask_volume = ask_volume[:, :-1]
    current_ask_volume = ask_volume[:, 1:]

    previous_mid = (previous_bid_price + previous_ask_price) / 2.0
    current_mid = (current_bid_price + current_ask_price) / 2.0
    previous_quote_valid = (
        (previous_mid >= 0.1)
        & (previous_ask_price >= previous_bid_price)
        & ((previous_ask_price - previous_bid_price) / previous_mid <= 0.1)
    )
    current_quote_valid = (
        (current_mid >= 0.1)
        & (current_ask_price >= current_bid_price)
        & ((current_ask_price - current_bid_price) / current_mid <= 0.1)
    )
    valid = (
        row_mask[:, :-1]
        & row_mask[:, 1:]
        & previous_quote_valid
        & current_quote_valid
        & (previous_bid_volume > 0)
        & (current_bid_volume > 0)
        & (previous_ask_volume > 0)
        & (current_ask_volume > 0)
    )
    bid_flow = np.where(current_bid_price >= previous_bid_price, current_bid_volume, 0.0)
    bid_flow -= np.where(current_bid_price <= previous_bid_price, previous_bid_volume, 0.0)
    ask_flow = np.where(current_ask_price <= previous_ask_price, current_ask_volume, 0.0)
    ask_flow -= np.where(current_ask_price >= previous_ask_price, previous_ask_volume, 0.0)
    depth = (
        previous_bid_volume
        + current_bid_volume
        + previous_ask_volume
        + current_ask_volume
    )
    valid &= np.isfinite(depth) & (depth > 0)
    ofi = np.divide(
        bid_flow - ask_flow,
        depth,
        out=np.zeros_like(depth, dtype=np.float64),
        where=valid,
    )
    return ofi, valid


def unscale(cache_block: np.ndarray, name: str, statistics: dict) -> np.ndarray:
    """Reverse the cache's training-only standardization for one source field."""

    values = cache_block[:, CHANNELS[name], :].astype(np.float64)
    values *= statistics[name]["scale"]
    values += statistics[name]["mean"]
    return values


def build_batch(cache_block: np.ndarray, statistics: dict) -> dict[str, np.ndarray]:
    """Create all target-free feature columns for one sample batch."""

    row_mask = cache_block[:, CHANNELS["row_mask"], :] > 0.5
    seconds = cache_block[:, CHANNELS["seconds_before_predict"], :].astype(np.float64) * 600.0
    window_180 = row_mask & (seconds <= 180.0)
    window_60 = row_mask & (seconds <= 60.0)
    window_20 = row_mask & (seconds <= 20.0)

    ask_1 = unscale(cache_block, "ask_price_1", statistics)
    bid_1 = unscale(cache_block, "bid_price_1", statistics)
    ask_volume_1 = unscale(cache_block, "ask_volume_1", statistics)
    bid_volume_1 = unscale(cache_block, "bid_volume_1", statistics)
    ask_2 = unscale(cache_block, "ask_price_2", statistics)
    bid_2 = unscale(cache_block, "bid_price_2", statistics)
    ask_volume_2 = unscale(cache_block, "ask_volume_2", statistics)
    bid_volume_2 = unscale(cache_block, "bid_volume_2", statistics)
    trade_price = unscale(cache_block, "transaction_avgprice", statistics)
    has_trade = (cache_block[:, CHANNELS["has_transactions"], :] > 0.5) & row_mask

    mid = (ask_1 + bid_1) / 2.0
    relative_spread_raw = np.divide(
        ask_1 - bid_1,
        mid,
        out=np.full_like(mid, np.inf),
        where=mid != 0,
    )
    # 零价格和明显交叉/异常盘口是数据哨兵，不应被解释为真实收益。
    # Zero prices and strongly crossed quotes are sentinels, not genuine returns.
    valid_mid = (
        row_mask
        & np.isfinite(mid)
        & (mid >= 0.1)
        & (relative_spread_raw >= 0.0)
        & (relative_spread_raw <= 0.1)
    )
    log_mid = np.log(np.where(valid_mid, mid, 1.0))
    return_mask = valid_mid[:, 1:] & valid_mid[:, :-1]
    log_returns = log_mid[:, 1:] - log_mid[:, :-1]
    return_seconds = seconds[:, 1:]
    return_180 = return_mask & (return_seconds <= 180.0)
    return_60 = return_mask & (return_seconds <= 60.0)
    return_20 = return_mask & (return_seconds <= 20.0)

    return_std, return_skew, return_kurtosis = return_moments(log_returns, return_mask)
    price_features = {
        "mid_return_full": endpoint_return(mid, valid_mid),
        "mid_return_180": endpoint_return(mid, valid_mid & window_180),
        "mid_return_60": endpoint_return(mid, valid_mid & window_60),
        "mid_return_20": endpoint_return(mid, valid_mid & window_20),
        "realized_volatility_full": np.sqrt(masked_sum(np.square(log_returns), return_mask)),
        "realized_volatility_180": np.sqrt(masked_sum(np.square(log_returns), return_180)),
        "realized_volatility_60": np.sqrt(masked_sum(np.square(log_returns), return_60)),
        "realized_volatility_20": np.sqrt(masked_sum(np.square(log_returns), return_20)),
        "mid_return_std_full": return_std,
        "mid_return_skew_full": return_skew,
        "mid_return_kurtosis_full": return_kurtosis,
        "mid_momentum_full": masked_slope(log_mid, -seconds, valid_mid),
        "mid_momentum_60": masked_slope(log_mid, -seconds, valid_mid & window_60),
    }
    mid_last = masked_last(mid, valid_mid)
    mid_full_mean = masked_mean(mid, valid_mid)
    mid_60_mean = masked_mean(mid, valid_mid & window_60)
    price_features["mid_last_vs_full_mean"] = (mid_last - mid_full_mean) / mid_full_mean
    price_features["mid_last_vs_60_mean"] = (mid_last - mid_60_mean) / mid_60_mean

    relative_spread = np.divide(
        ask_1 - bid_1,
        mid,
        out=np.zeros_like(mid),
        where=valid_mid,
    )
    price_features.update(
        {
            "relative_spread_mean": masked_mean(relative_spread, valid_mid),
            "relative_spread_std": masked_std(relative_spread, valid_mid),
            "relative_spread_last": masked_last(relative_spread, valid_mid),
            "relative_spread_60_mean": masked_mean(relative_spread, valid_mid & window_60),
        }
    )
    trade_deviation = np.divide(
        trade_price - mid,
        mid,
        out=np.zeros_like(mid),
        where=has_trade & valid_mid,
    )
    trade_mask = (
        has_trade
        & valid_mid
        & np.isfinite(trade_price)
        & (trade_price >= 0.1)
        & (np.abs(trade_deviation) <= 0.1)
    )
    price_features["trade_price_vs_mid_mean"] = masked_mean(trade_deviation, trade_mask)
    price_features["trade_price_vs_mid_last"] = masked_last(trade_deviation, trade_mask)

    valid_depth_1 = row_mask & (ask_volume_1 > 0) & (bid_volume_1 > 0)
    valid_depth_2 = row_mask & (ask_volume_2 > 0) & (bid_volume_2 > 0)
    depth_1 = ask_volume_1 + bid_volume_1
    depth_2 = ask_volume_2 + bid_volume_2
    imbalance_1 = np.divide(
        bid_volume_1 - ask_volume_1,
        depth_1,
        out=np.zeros_like(depth_1),
        where=valid_depth_1,
    )
    imbalance_2 = np.divide(
        bid_volume_2 - ask_volume_2,
        depth_2,
        out=np.zeros_like(depth_2),
        where=valid_depth_2,
    )
    valid_total_depth = valid_depth_1 & valid_depth_2
    total_depth = depth_1 + depth_2
    total_imbalance = np.divide(
        bid_volume_1 + bid_volume_2 - ask_volume_1 - ask_volume_2,
        total_depth,
        out=np.zeros_like(total_depth),
        where=valid_total_depth,
    )
    microprice = np.divide(
        ask_1 * bid_volume_1 + bid_1 * ask_volume_1,
        depth_1,
        out=np.zeros_like(mid),
        where=valid_depth_1 & valid_mid,
    )
    microprice_displacement = np.divide(
        microprice - mid,
        mid,
        out=np.zeros_like(mid),
        where=valid_depth_1 & valid_mid,
    )
    micro_mask = valid_depth_1 & valid_mid

    ofi_1, ofi_mask_1 = normalized_ofi(
        bid_1, bid_volume_1, ask_1, ask_volume_1, row_mask
    )
    ofi_2, ofi_mask_2 = normalized_ofi(
        bid_2, bid_volume_2, ask_2, ask_volume_2, row_mask
    )
    event_seconds = seconds[:, 1:]
    ofi_1_60 = ofi_mask_1 & (event_seconds <= 60.0)
    ofi_1_20 = ofi_mask_1 & (event_seconds <= 20.0)
    ofi_1_previous = ofi_mask_1 & (event_seconds > 20.0) & (event_seconds <= 60.0)
    ofi_2_60 = ofi_mask_2 & (event_seconds <= 60.0)
    multilevel_mask = ofi_mask_1 & ofi_mask_2
    multilevel_ofi = ofi_1 + ofi_2
    multilevel_60 = multilevel_mask & (event_seconds <= 60.0)
    multilevel_20 = multilevel_mask & (event_seconds <= 20.0)
    multilevel_previous = multilevel_mask & (event_seconds > 20.0) & (event_seconds <= 60.0)

    log_total_depth = np.log1p(np.where(valid_total_depth, total_depth, 0.0))
    book_features = {
        "book_imbalance_1_mean_robust": masked_mean(imbalance_1, valid_depth_1),
        "book_imbalance_1_std_robust": masked_std(imbalance_1, valid_depth_1),
        "book_imbalance_1_last_robust": masked_last(imbalance_1, valid_depth_1),
        "book_imbalance_1_60_mean_robust": masked_mean(imbalance_1, valid_depth_1 & window_60),
        "book_imbalance_1_20_mean_robust": masked_mean(imbalance_1, valid_depth_1 & window_20),
        "book_imbalance_2_mean_robust": masked_mean(imbalance_2, valid_depth_2),
        "book_imbalance_2_std_robust": masked_std(imbalance_2, valid_depth_2),
        "book_imbalance_2_last_robust": masked_last(imbalance_2, valid_depth_2),
        "book_imbalance_2_60_mean_robust": masked_mean(imbalance_2, valid_depth_2 & window_60),
        "total_book_imbalance_mean": masked_mean(total_imbalance, valid_total_depth),
        "total_book_imbalance_last": masked_last(total_imbalance, valid_total_depth),
        "total_book_imbalance_60_mean": masked_mean(total_imbalance, valid_total_depth & window_60),
        "microprice_displacement_mean": masked_mean(microprice_displacement, micro_mask),
        "microprice_displacement_std": masked_std(microprice_displacement, micro_mask),
        "microprice_displacement_last": masked_last(microprice_displacement, micro_mask),
        "microprice_displacement_60_mean": masked_mean(microprice_displacement, micro_mask & window_60),
        "microprice_displacement_20_mean": masked_mean(microprice_displacement, micro_mask & window_20),
        "ofi_1_mean": masked_mean(ofi_1, ofi_mask_1),
        "ofi_1_std": masked_std(ofi_1, ofi_mask_1),
        "ofi_1_sum": masked_sum(ofi_1, ofi_mask_1),
        "ofi_1_last": masked_last(ofi_1, ofi_mask_1),
        "ofi_1_60_mean": masked_mean(ofi_1, ofi_1_60),
        "ofi_1_20_mean": masked_mean(ofi_1, ofi_1_20),
        "ofi_1_acceleration": masked_mean(ofi_1, ofi_1_20) - masked_mean(ofi_1, ofi_1_previous),
        "ofi_2_mean": masked_mean(ofi_2, ofi_mask_2),
        "ofi_2_sum": masked_sum(ofi_2, ofi_mask_2),
        "ofi_2_60_mean": masked_mean(ofi_2, ofi_2_60),
        "multilevel_ofi_mean": masked_mean(multilevel_ofi, multilevel_mask),
        "multilevel_ofi_sum": masked_sum(multilevel_ofi, multilevel_mask),
        "multilevel_ofi_60_mean": masked_mean(multilevel_ofi, multilevel_60),
        "multilevel_ofi_acceleration": masked_mean(multilevel_ofi, multilevel_20)
        - masked_mean(multilevel_ofi, multilevel_previous),
        "log_total_depth_mean": masked_mean(log_total_depth, valid_total_depth),
        "log_total_depth_std": masked_std(log_total_depth, valid_total_depth),
        "log_total_depth_last": masked_last(log_total_depth, valid_total_depth),
        "log_total_depth_60_mean": masked_mean(log_total_depth, valid_total_depth & window_60),
    }
    return {**price_features, **book_features}


def main() -> None:
    """Build, validate, and save the compact feature table."""

    project_dir = Path(__file__).resolve().parents[1]
    cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    output_path = project_dir / "data" / "processed" / "train_market_microstructure_features.feather"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    feature_columns = PRICE_FEATURE_COLUMNS + BOOK_FEATURE_COLUMNS
    output = np.empty((cache.shape[0], len(feature_columns)), dtype=np.float32)

    statistics = {}
    for name in CHANNELS:
        statistics_path = cache_dir / "statistics" / f"{name}.json"
        if statistics_path.exists():
            statistics[name] = json.loads(statistics_path.read_text(encoding="utf-8"))

    for start in range(0, cache.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, cache.shape[0])
        batch_features = build_batch(cache[start:end], statistics)
        if list(batch_features) != feature_columns:
            raise AssertionError("Feature construction order does not match the declared schema.")
        for column_index, column_name in enumerate(feature_columns):
            output[start:end, column_index] = batch_features[column_name]
        if start % 100_000 == 0:
            print(f"processed samples {start}:{end}")

    if np.isinf(output).any():
        raise ValueError("Microstructure features contain infinity.")
    feature_data = pd.DataFrame(output, columns=feature_columns)
    feature_data.insert(0, "sample_id", np.arange(len(feature_data), dtype=np.int32))

    # 先写入纯英文临时路径，规避底层 Arrow 在 Windows 中文路径上的兼容问题。
    # Write to an ASCII temporary path first for reliable Arrow behavior on Windows.
    temporary_file = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        feature_data.to_feather(temporary_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"microstructure feature shape = {feature_data.shape}")
    print(f"price feature count = {len(PRICE_FEATURE_COLUMNS)}")
    print(f"book feature count = {len(BOOK_FEATURE_COLUMNS)}")
    print(f"output = {output_path}")


if __name__ == "__main__":
    main()
