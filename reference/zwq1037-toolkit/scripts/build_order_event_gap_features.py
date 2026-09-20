"""Build order-event timing dispersion features in native file order."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from build_event_flow_features import master_sample_ids, safe_write_ipc


FEATURE_COLUMNS = [
    "order_event_gap_mean",
    "order_event_gap_std",
    "order_event_gap_max",
    "order_event_nonzero_gap_share",
]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    input_path = project_dir / "data" / "raw" / "train" / "order.feather"
    output_path = (
        project_dir / "data" / "processed" / "train_order_event_gap_features.feather"
    )
    # 原始订单文件已经按sample_id升序、时间倒序排列。
    # The raw order file is ordered by sample id and descending event time.
    events = (
        pl.scan_ipc(input_path)
        .select("sample_id", "seconds_before_predict")
        .with_columns(
            pl.col("seconds_before_predict")
            .diff()
            .over("sample_id")
            .abs()
            .alias("event_gap")
        )
    )
    aggregates = (
        events.group_by("sample_id")
        .agg(
            pl.col("event_gap").mean().alias("order_event_gap_mean"),
            pl.col("event_gap").std().alias("order_event_gap_std"),
            pl.col("event_gap").max().alias("order_event_gap_max"),
            (pl.col("event_gap") > 0).mean().alias("order_event_nonzero_gap_share"),
        )
        .collect(engine="streaming")
    )
    features = master_sample_ids(project_dir, "train").join(
        aggregates, on="sample_id", how="left", validate="1:1"
    ).select("sample_id", *FEATURE_COLUMNS)
    if features.height != 1_257_637 or features["sample_id"].n_unique() != features.height:
        raise AssertionError("Malformed order event-gap feature table.")
    safe_write_ipc(features, output_path)
    print(f"feature_shape={features.shape}", flush=True)
    print(f"null_counts={features.null_count().to_dicts()[0]}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
