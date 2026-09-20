"""Build deployable transaction activity and inactivity features on a 61-second grid."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from build_event_flow_features import master_sample_ids, safe_write_ipc


FEATURE_COLUMNS = [
    "trade_active_second_ratio_60",
    "trade_active_second_ratio_30",
    "trade_active_second_ratio_10",
    "trade_max_inactive_run_ratio_60",
]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    input_path = project_dir / "data" / "raw" / "train" / "transaction.feather"
    output_path = (
        project_dir
        / "data"
        / "processed"
        / "train_transaction_second_activity_features.feather"
    )
    # 与公开Notebook一致，把0～59.999秒四舍五入到0～60的61个格子。
    # Match the public notebook's rounding to a 61-bin grid from second 0 through 60.
    active_seconds = (
        pl.scan_ipc(input_path)
        .select(
            "sample_id",
            pl.col("seconds_before_predict")
            .round(0)
            .cast(pl.Int16)
            .alias("second"),
        )
        .group_by("sample_id", "second")
        .agg(pl.len().alias("events_in_second"))
    )
    aggregates = (
        active_seconds.group_by("sample_id")
        .agg(
            pl.len().cast(pl.Int16).alias("active_second_count_60"),
            (pl.col("second") <= 30)
            .sum()
            .cast(pl.Int16)
            .alias("active_second_count_30"),
            (pl.col("second") <= 10)
            .sum()
            .cast(pl.Int16)
            .alias("active_second_count_10"),
            (pl.col("second").sort().diff() - 1)
            .max()
            .fill_null(0)
            .alias("internal_inactive_run"),
            pl.col("second").min().alias("minimum_active_second"),
            pl.col("second").max().alias("maximum_active_second"),
        )
        .collect(engine="streaming")
    )
    aggregates = master_sample_ids(project_dir, "train").join(
        aggregates, on="sample_id", how="left", validate="1:1"
    )
    features = aggregates.with_columns(
        pl.col("active_second_count_60").fill_null(0),
        pl.col("active_second_count_30").fill_null(0),
        pl.col("active_second_count_10").fill_null(0),
    ).with_columns(
        (pl.col("active_second_count_60") / 61.0).alias(
            "trade_active_second_ratio_60"
        ),
        (pl.col("active_second_count_30") / 31.0).alias(
            "trade_active_second_ratio_30"
        ),
        (pl.col("active_second_count_10") / 11.0).alias(
            "trade_active_second_ratio_10"
        ),
        pl.when(pl.col("active_second_count_60") > 0)
        .then(
            pl.max_horizontal(
                pl.col("internal_inactive_run").fill_null(0),
                pl.col("minimum_active_second"),
                60 - pl.col("maximum_active_second"),
            )
            / 61.0
        )
        .otherwise(1.0)
        .alias("trade_max_inactive_run_ratio_60"),
    ).select("sample_id", *FEATURE_COLUMNS)
    if features.height != 1_257_637 or features["sample_id"].n_unique() != features.height:
        raise AssertionError("Malformed transaction second-activity feature table.")
    if features.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise AssertionError("Unexpected nulls in transaction second-activity features.")
    safe_write_ipc(features, output_path)
    print(f"feature_shape={features.shape}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
