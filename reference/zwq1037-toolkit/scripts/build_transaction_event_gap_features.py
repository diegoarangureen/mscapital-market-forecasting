"""Build transaction-event timing dispersion features in native file order."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from build_event_flow_features import master_sample_ids, safe_write_ipc


FEATURE_COLUMNS = [
    "trade_event_gap_mean",
    "trade_event_gap_std",
    "trade_event_gap_max",
    "trade_event_nonzero_gap_share",
]


def parse_arguments() -> argparse.Namespace:
    """Choose whether to build training or test features."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    split = arguments.split
    project_dir = Path(__file__).resolve().parents[1]
    input_path = project_dir / "data" / "raw" / split / "transaction.feather"
    output_path = (
        project_dir
        / "data"
        / "processed"
        / f"{split}_transaction_event_gap_features.feather"
    )
    # 原始文件已经按sample_id升序、时间倒序排列，先前已做全文件验证。
    # The raw file is ordered by sample id and descending event time, verified globally.
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
            pl.col("event_gap").mean().alias("trade_event_gap_mean"),
            pl.col("event_gap").std().alias("trade_event_gap_std"),
            pl.col("event_gap").max().alias("trade_event_gap_max"),
            (pl.col("event_gap") > 0)
            .mean()
            .alias("trade_event_nonzero_gap_share"),
        )
        .collect(engine="streaming")
    )
    expected_ids = master_sample_ids(project_dir, split)
    features = expected_ids.join(
        aggregates, on="sample_id", how="left", validate="1:1"
    ).select("sample_id", *FEATURE_COLUMNS)
    if (
        features.height != expected_ids.height
        or features["sample_id"].n_unique() != features.height
    ):
        raise AssertionError("Malformed transaction event-gap feature table.")
    safe_write_ipc(features, output_path)
    print(f"feature_shape={features.shape}", flush=True)
    print(f"null_counts={features.null_count().to_dicts()[0]}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
