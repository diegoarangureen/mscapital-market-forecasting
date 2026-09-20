"""Build leakage-safe large-transaction features for the formal validation split."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from build_event_flow_features import master_sample_ids, safe_ratio, safe_write_ipc


FEATURE_COLUMNS = [
    "large_trade_count_imbalance_q90",
    "large_trade_volume_imbalance_q90",
    "large_trade_count_share_q90",
    "large_trade_volume_share_q90",
]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    transaction_path = project_dir / "data" / "raw" / "train" / "transaction.feather"
    label_path = project_dir / "data" / "raw" / "label.feather"
    output_path = (
        project_dir
        / "data"
        / "processed"
        / "train_transaction_large_trade_q90_features.feather"
    )
    metadata_path = (
        project_dir
        / "data"
        / "interim"
        / "transaction_large_trade_q90_validation_metadata.json"
    )

    events = pl.scan_ipc(transaction_path).select(
        "sample_id", "volume", "side"
    )
    fitting_ids = (
        pl.scan_ipc(label_path)
        .filter(pl.col("month") <= 59)
        .select("sample_id")
    )
    # 只使用训练月份拟合阈值，再固定应用到验证月份。
    # Fit the threshold on training months only, then freeze it for validation.
    threshold = (
        events.join(fitting_ids, on="sample_id", how="inner")
        .select(pl.col("volume").quantile(0.90).alias("threshold"))
        .collect(engine="streaming")
        .item()
    )
    if threshold is None or threshold <= 0:
        raise AssertionError(f"Invalid q90 threshold: {threshold}")

    is_large = pl.col("volume") > threshold
    is_buy = pl.col("side") == 0
    is_sell = pl.col("side") == 1
    aggregates = (
        events.group_by("sample_id")
        .agg(
            (is_large & is_buy).cast(pl.Int64).sum().alias("large_buy_count"),
            (is_large & is_sell).cast(pl.Int64).sum().alias("large_sell_count"),
            pl.when(is_large & is_buy)
            .then(pl.col("volume").cast(pl.Int64))
            .otherwise(0)
            .sum()
            .alias("large_buy_volume"),
            pl.when(is_large & is_sell)
            .then(pl.col("volume").cast(pl.Int64))
            .otherwise(0)
            .sum()
            .alias("large_sell_volume"),
            pl.len().cast(pl.Int64).alias("total_trade_count"),
            pl.col("volume").cast(pl.Int64).sum().alias("total_trade_volume"),
        )
        .collect(engine="streaming")
    )
    aggregates = master_sample_ids(project_dir, "train").join(
        aggregates, on="sample_id", how="left", validate="1:1"
    )
    value_columns = [column for column in aggregates.columns if column != "sample_id"]
    aggregates = aggregates.with_columns(pl.col(value_columns).fill_null(0))
    large_count = pl.col("large_buy_count") + pl.col("large_sell_count")
    large_volume = pl.col("large_buy_volume") + pl.col("large_sell_volume")
    features = aggregates.with_columns(
        safe_ratio(
            pl.col("large_buy_count") - pl.col("large_sell_count"), large_count
        ).alias("large_trade_count_imbalance_q90"),
        safe_ratio(
            pl.col("large_buy_volume") - pl.col("large_sell_volume"), large_volume
        ).alias("large_trade_volume_imbalance_q90"),
        safe_ratio(large_count, pl.col("total_trade_count")).alias(
            "large_trade_count_share_q90"
        ),
        safe_ratio(large_volume, pl.col("total_trade_volume")).alias(
            "large_trade_volume_share_q90"
        ),
    ).select("sample_id", *FEATURE_COLUMNS)
    if features.height != 1_257_637 or features["sample_id"].n_unique() != features.height:
        raise AssertionError("Malformed large-trade feature table.")
    safe_write_ipc(features, output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "quantile": 0.90,
                "strictly_greater_than_threshold": True,
                "threshold_fitted_on_months": "0-59",
                "volume_threshold": int(threshold),
                "feature_columns": FEATURE_COLUMNS,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"q90_volume_threshold={threshold}", flush=True)
    print(f"feature_shape={features.shape}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
