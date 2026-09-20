"""Build sixteen target-free three-stream lead/lag response features.

The final 60 seconds of each sample are split into twenty chronological
three-second bins.  All signals and responses end before the prediction time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl


N_BINS = 20
BIN_SECONDS = 3.0
FEATURE_COLUMNS = [
    f"lag_{signal}_return_{lag}bin"
    for signal in ("trade", "order", "ofi")
    for lag in (0, 1, 2)
] + [
    "lag_trade_next_return_slope",
    "lag_order_next_return_slope",
    "lag_ofi_next_return_slope",
    "lag_buy_no_rise_share",
    "lag_sell_no_fall_share",
    "lag_trade_order_replenish_1bin",
    "lag_trade_order_replenish_2bin",
]


def bin_expr() -> pl.Expr:
    """Return 0 for the oldest bin and 19 for the newest bin."""

    return (
        pl.lit(N_BINS - 1)
        - (pl.col("seconds_before_predict") / BIN_SECONDS)
        .floor()
        .cast(pl.Int16)
    ).cast(pl.Int8)


def scan_recent(path: Path, max_sample_id: int | None) -> pl.LazyFrame:
    """Project only observations inside the fully observed 60-second window."""

    frame = pl.scan_ipc(path).filter(
        (pl.col("seconds_before_predict") >= 0.0)
        & (pl.col("seconds_before_predict") < 60.0)
    )
    if max_sample_id is not None:
        frame = frame.filter(pl.col("sample_id") < max_sample_id)
    return frame.with_columns(bin_expr().alias("bin"))


def dense_column(
    grouped: pl.DataFrame, column: str, row_count: int, fill: float = 0.0
) -> np.ndarray:
    """Place sparse sample/bin aggregates in a dense chronological grid."""

    result = np.full((row_count, N_BINS), fill, dtype=np.float32)
    row = grouped["sample_id"].to_numpy().astype(np.int64, copy=False)
    col = grouped["bin"].to_numpy().astype(np.int64, copy=False)
    if row_count and (row.min(initial=0) < 0 or row.max(initial=0) >= row_count):
        raise AssertionError("Sample ID outside dense-grid bounds.")
    result[row, col] = grouped[column].to_numpy().astype(np.float32, copy=False)
    return result


def load_market_grid(raw_dir: Path, row_count: int, max_sample_id: int | None) -> dict:
    """Aggregate the last visible quote in each three-second bin."""

    frame = scan_recent(raw_dir / "market.feather", max_sample_id).select(
        "sample_id",
        "bin",
        "seconds_before_predict",
        "ask_price_1",
        "bid_price_1",
        "ask_volume_1",
        "bid_volume_1",
    )
    grouped = frame.group_by("sample_id", "bin").agg(
        pl.col("ask_price_1")
        .sort_by("seconds_before_predict", descending=True)
        .last()
        .alias("ask_price"),
        pl.col("bid_price_1")
        .sort_by("seconds_before_predict", descending=True)
        .last()
        .alias("bid_price"),
        pl.col("ask_volume_1")
        .sort_by("seconds_before_predict", descending=True)
        .last()
        .alias("ask_volume"),
        pl.col("bid_volume_1")
        .sort_by("seconds_before_predict", descending=True)
        .last()
        .alias("bid_volume"),
    ).collect(engine="streaming")
    result = {
        name: dense_column(grouped, name, row_count, np.nan)
        for name in ("ask_price", "bid_price", "ask_volume", "bid_volume")
    }
    del grouped
    # A missing three-second bin carries the last older observed quote forward.
    # 某个3秒格没有快照时，沿用此前已观测到的报价。
    for name, grid in result.items():
        for index in range(1, N_BINS):
            missing = ~np.isfinite(grid[:, index])
            grid[missing, index] = grid[missing, index - 1]
    return result


def load_flow_grid(
    raw_dir: Path,
    stream: str,
    row_count: int,
    max_sample_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate signed and gross event volume per chronological bin."""

    frame = scan_recent(raw_dir / f"{stream}.feather", max_sample_id)
    if stream == "transaction":
        frame = frame.select("sample_id", "bin", "volume", "side").with_columns(
            pl.when(pl.col("side") == 0)
            .then(1.0)
            .when(pl.col("side") == 1)
            .then(-1.0)
            .otherwise(0.0)
            .alias("sign")
        )
    elif stream == "order":
        frame = frame.select("sample_id", "bin", "volume", "side", "order_action").with_columns(
            (
                pl.when(pl.col("side") == 0)
                .then(1.0)
                .when(pl.col("side") == 1)
                .then(-1.0)
                .otherwise(0.0)
                * pl.when(pl.col("order_action") == 0)
                .then(1.0)
                .when(pl.col("order_action") == 1)
                .then(-1.0)
                .otherwise(0.0)
            ).alias("sign")
        )
    else:
        raise ValueError(stream)
    grouped = frame.group_by("sample_id", "bin").agg(
        (pl.col("volume") * pl.col("sign")).sum().alias("signed_volume"),
        pl.col("volume").sum().alias("gross_volume"),
    ).collect(engine="streaming")
    signed = dense_column(grouped, "signed_volume", row_count)
    gross = dense_column(grouped, "gross_volume", row_count)
    return signed, gross


def normalized_inner(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-sample normalized inner products across observed bins."""

    numerator = np.sum(a * b, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(a * a, axis=1, dtype=np.float64)
        * np.sum(b * b, axis=1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-8,
    ).astype(np.float32)


def response_slope(a: np.ndarray, next_return: np.ndarray) -> np.ndarray:
    """Compute a stable one-bin forward response slope."""

    numerator = np.sum(a[:, :-1] * next_return[:, 1:], axis=1, dtype=np.float64)
    denominator = np.sum(a[:, :-1] ** 2, axis=1, dtype=np.float64) + 0.01
    return (numerator / denominator).astype(np.float32)


def build_features(project_dir: Path, split: str, max_sample_id: int | None) -> pl.DataFrame:
    """Build all sixteen lag features using only the three raw input streams."""

    raw_dir = project_dir / "data" / "raw" / split
    if split == "train":
        ids = pl.read_ipc(
            project_dir / "data" / "raw" / "label.feather",
            columns=["sample_id"],
        )["sample_id"].to_numpy()
    else:
        ids = pl.scan_ipc(raw_dir / "market.feather").select("sample_id").unique().collect(
            engine="streaming"
        )["sample_id"].to_numpy()
    if max_sample_id is not None:
        ids = ids[ids < max_sample_id]
    ids = np.sort(np.unique(ids)).astype(np.int32)
    if not np.array_equal(ids, np.arange(len(ids), dtype=np.int32)):
        raise AssertionError("Expected contiguous sample IDs beginning at zero.")
    row_count = len(ids)

    market = load_market_grid(raw_dir, row_count, max_sample_id)
    ask = market["ask_price"]
    bid = market["bid_price"]
    ask_volume = market["ask_volume"]
    bid_volume = market["bid_volume"]
    quote_valid = np.isfinite(ask) & np.isfinite(bid) & (ask > 0) & (bid > 0) & (ask >= bid)
    depth = np.where(quote_valid, ask_volume + bid_volume, np.nan)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    mid = np.where(quote_valid, 0.5 * (ask + bid), np.nan)
    spread = np.where(quote_valid, ask - bid, np.nan)
    returns = np.zeros((row_count, N_BINS), dtype=np.float32)
    valid_pair = quote_valid[:, 1:] & quote_valid[:, :-1]
    ratio = (mid[:, 1:] - mid[:, :-1]) / np.maximum(spread[:, :-1], 1.0e-8)
    returns[:, 1:] = np.where(valid_pair, np.clip(ratio, -5.0, 5.0), 0.0)

    bid_up = bid[:, 1:] >= bid[:, :-1]
    bid_down = bid[:, 1:] <= bid[:, :-1]
    ask_down = ask[:, 1:] <= ask[:, :-1]
    ask_up = ask[:, 1:] >= ask[:, :-1]
    ofi_raw = (
        np.where(bid_up, bid_volume[:, 1:], 0.0)
        - np.where(bid_down, bid_volume[:, :-1], 0.0)
        - np.where(ask_down, ask_volume[:, 1:], 0.0)
        + np.where(ask_up, ask_volume[:, :-1], 0.0)
    )
    ofi = np.zeros((row_count, N_BINS), dtype=np.float32)
    ofi[:, 1:] = np.where(
        valid_pair,
        np.clip(ofi_raw / np.maximum(depth[:, 1:] + depth[:, :-1], 1.0), -5.0, 5.0),
        0.0,
    )
    del market, ask, bid, ask_volume, bid_volume, mid, spread, ofi_raw

    tx_signed, tx_gross = load_flow_grid(raw_dir, "transaction", row_count, max_sample_id)
    flow_signed, flow_gross = load_flow_grid(raw_dir, "order", row_count, max_sample_id)
    trade = np.clip(tx_signed / np.maximum(depth + tx_gross, 1.0), -5.0, 5.0)
    order = np.clip(flow_signed / np.maximum(depth + flow_gross, 1.0), -5.0, 5.0)
    del tx_signed, tx_gross, flow_signed, flow_gross, depth

    signals = {"trade": trade, "order": order, "ofi": ofi}
    output: dict[str, np.ndarray] = {}
    for signal_name, signal in signals.items():
        for lag in (0, 1, 2):
            left = signal[:, : N_BINS - lag]
            right = returns[:, lag:]
            output[f"lag_{signal_name}_return_{lag}bin"] = normalized_inner(
                left, right
            )
        output[f"lag_{signal_name}_next_return_slope"] = response_slope(
            signal, returns
        )

    buy_pressure = np.clip(trade + order + ofi, 0.0, None)
    sell_pressure = np.clip(-(trade + order + ofi), 0.0, None)
    buy_no_rise = np.sum(
        buy_pressure[:, :-1] * (returns[:, 1:] <= 0.0), axis=1, dtype=np.float64
    )
    sell_no_fall = np.sum(
        sell_pressure[:, :-1] * (returns[:, 1:] >= 0.0), axis=1, dtype=np.float64
    )
    output["lag_buy_no_rise_share"] = (
        buy_no_rise / np.maximum(np.sum(buy_pressure[:, :-1], axis=1), 1.0e-8)
    ).astype(np.float32)
    output["lag_sell_no_fall_share"] = (
        sell_no_fall / np.maximum(np.sum(sell_pressure[:, :-1], axis=1), 1.0e-8)
    ).astype(np.float32)
    for lag in (1, 2):
        output[f"lag_trade_order_replenish_{lag}bin"] = normalized_inner(
            trade[:, : N_BINS - lag], -order[:, lag:]
        )
    for values in output.values():
        values[~np.isfinite(values)] = 0.0
    return pl.DataFrame({"sample_id": ids, **{name: output[name] for name in FEATURE_COLUMNS}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--max-sample-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    output_path = arguments.output or (
        project_dir / "data" / "processed" / f"{arguments.split}_three_stream_lag_features.feather"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    features = build_features(project_dir, arguments.split, arguments.max_sample_id)
    if features.width != 17:
        raise AssertionError("Expected sixteen new features plus sample_id.")
    if not features.select(pl.exclude("sample_id").is_finite().all()).row(0) == tuple(
        [True] * 16
    ):
        raise AssertionError("Non-finite lag feature values.")
    features.write_ipc(output_path, compression="uncompressed")
    summary = {
        "split": arguments.split,
        "rows": features.height,
        "feature_count": 16,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
