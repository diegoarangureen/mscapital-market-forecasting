"""Build order-at-quote features for the older 15--60-second interval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from build_order_quote_position_features import (
    derive_window_features,
    load_market,
    load_orders,
    matched_orders,
    window_aggregations,
)


FEATURE_COLUMNS = [
    "order_crossable_new_direction_share_45",
    "order_inside_new_direction_share_45",
    "order_near_new_pressure_45",
    "order_near_cancel_pressure_45",
    "order_near_net_pressure_45",
    "order_cancel_ratio_side_diff_45",
    "order_new_closeness_side_diff_45",
    "order_depth_adjusted_net_rate_45",
    "order_quote_matched_volume_share_45",
]


def build_features(project_dir: Path, split: str, max_sample_id: int | None) -> pl.DataFrame:
    """Aggregate only orders observed between 15 and 60 seconds before prediction."""

    raw_dir = project_dir / "data" / "raw" / split
    market = load_market(raw_dir / "market.feather", max_sample_id)
    orders = load_orders(raw_dir / "order.feather", max_sample_id)
    matched = matched_orders(orders, market)
    older_window = (pl.col("order_seconds") > 15.0) & (pl.col("order_seconds") <= 60.0)
    grouped = matched.group_by("sample_id").agg(*window_aggregations(45, older_window))
    return (
        grouped.with_columns(*derive_window_features(45))
        .select("sample_id", *FEATURE_COLUMNS)
        .sort("sample_id")
        .collect(engine="streaming")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--max-sample-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    output_path = arguments.output or (
        project_dir / "data" / "processed" / f"{arguments.split}_order_quote_early45_features.feather"
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
