"""Build compact interactions between aggressive trades and visible L1 depth."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


CROSS_FEATURE_COLUMNS = [
    "aggressive_buy_to_ask_depth_log1p",
    "aggressive_sell_to_bid_depth_log1p",
    "aggressive_depth_pressure_difference",
    "trade_minus_book_imbalance_60",
    "trade_times_book_imbalance_60",
]


def safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return log1p(numerator / denominator) only for positive visible depth."""

    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator >= 0)
        & (denominator > 0)
    )
    result[valid] = np.log1p(numerator[valid] / denominator[valid])
    return result


def safe_write(data: pd.DataFrame, output_path: Path) -> None:
    """Write Feather through an ASCII temporary path on Windows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(suffix=".feather", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        data.to_feather(temporary_path)
        shutil.copyfile(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Build full train or test cross features; no smoke mode is provided."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    transaction = pd.read_feather(
        processed_dir / f"{arguments.split}_transaction_flow_features.feather"
    )
    recent = pd.read_feather(
        processed_dir / f"{arguments.split}_market_last60_features_complete.feather",
        columns=[
            "sample_id",
            "last60_ask_volume_1_mean",
            "last60_bid_volume_1_mean",
        ],
    )
    micro = pd.read_feather(
        processed_dir / f"{arguments.split}_market_microstructure_features.feather",
        columns=["sample_id", "book_imbalance_1_60_mean_robust"],
    )
    data = (
        transaction.merge(recent, on="sample_id", how="left", validate="one_to_one")
        .merge(micro, on="sample_id", how="left", validate="one_to_one")
    )
    total_volume = np.expm1(data["log1p_total_trade_volume_60"].to_numpy(dtype=np.float64))
    trade_imbalance = data["trade_volume_imbalance_60"].to_numpy(dtype=np.float64)
    buy_volume = total_volume * (1.0 + trade_imbalance) / 2.0
    sell_volume = total_volume * (1.0 - trade_imbalance) / 2.0
    buy_pressure = safe_log_ratio(
        buy_volume, data["last60_ask_volume_1_mean"].to_numpy(dtype=np.float64)
    )
    sell_pressure = safe_log_ratio(
        sell_volume, data["last60_bid_volume_1_mean"].to_numpy(dtype=np.float64)
    )
    book_imbalance = data["book_imbalance_1_60_mean_robust"].to_numpy(dtype=np.float64)
    output = pd.DataFrame(
        {
            "sample_id": data["sample_id"].to_numpy(dtype=np.int32),
            "aggressive_buy_to_ask_depth_log1p": buy_pressure.astype(np.float32),
            "aggressive_sell_to_bid_depth_log1p": sell_pressure.astype(np.float32),
            "aggressive_depth_pressure_difference": (buy_pressure - sell_pressure).astype(np.float32),
            "trade_minus_book_imbalance_60": (trade_imbalance - book_imbalance).astype(np.float32),
            "trade_times_book_imbalance_60": (trade_imbalance * book_imbalance).astype(np.float32),
        }
    )
    if np.isinf(output[CROSS_FEATURE_COLUMNS].to_numpy(dtype=np.float64)).any():
        raise AssertionError("Cross features contain infinity.")
    output_path = processed_dir / f"{arguments.split}_event_market_cross_features.feather"
    safe_write(output, output_path)
    print(f"event-market cross feature shape = {output.shape}")
    print(f"output = {output_path}")


if __name__ == "__main__":
    main()
