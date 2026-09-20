"""Create a tiny market sample without loading all market columns together."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl


MARKET_COLUMNS = [
    "sample_id",
    "seconds_before_predict",
    "transaction_avgprice",
    "transaction_volume",
    "transaction_count",
    "ask_price_1",
    "ask_volume_1",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "bid_price_2",
    "bid_volume_2",
]


def extract_one_column(
    market_path: Path,
    column_name: str,
    row_count: int,
    output_path: Path,
) -> None:
    """Read one compressed column in a short-lived process and save its prefix."""

    column_data = pl.read_ipc(
        market_path,
        columns=[column_name],
        n_rows=row_count,
        memory_map=False,
    )
    column_data.write_ipc(output_path)


def build_sample(project_dir: Path) -> None:
    """Build market and label samples for sample IDs 0 through 4."""

    market_path = project_dir / "data" / "raw" / "train" / "market.feather"
    label_path = project_dir / "data" / "raw" / "label.feather"
    interim_dir = project_dir / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    selected_sample_ids = [0, 1, 2, 3, 4]

    # Only the ID column is opened here. The process never holds all 13 columns.
    sample_id_prefix = pl.read_ipc(
        market_path,
        columns=["sample_id"],
        n_rows=10_000,
        memory_map=False,
    )

    if not sample_id_prefix["sample_id"].is_sorted():
        raise RuntimeError("The first market rows are not sorted by sample_id.")

    selected_mask = sample_id_prefix["sample_id"].is_in(selected_sample_ids)
    selected_row_count = int(selected_mask.sum())
    next_sample_id = int(sample_id_prefix[selected_row_count, "sample_id"])

    if next_sample_id in selected_sample_ids:
        raise RuntimeError("The prefix ended before the selected samples were complete.")

    with tempfile.TemporaryDirectory(prefix="lesson_03b_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        small_column_paths = []

        for column_index, column_name in enumerate(MARKET_COLUMNS):
            small_column_path = temporary_path / f"{column_index:02d}_{column_name}.feather"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--extract-column",
                "--market-path",
                str(market_path),
                "--column-name",
                column_name,
                "--row-count",
                str(selected_row_count),
                "--output-path",
                str(small_column_path),
            ]
            subprocess.run(command, check=True)
            small_column_paths.append(small_column_path)

        small_columns = [pl.read_ipc(path) for path in small_column_paths]
        market_sample = pl.concat(small_columns, how="horizontal_extend")

    actual_ids = market_sample["sample_id"].unique(maintain_order=True).to_list()
    if actual_ids != selected_sample_ids:
        raise RuntimeError(f"Unexpected sample IDs: {actual_ids}")

    label_data = pl.read_ipc(label_path)
    label_sample = label_data.filter(pl.col("sample_id").is_in(selected_sample_ids))

    market_output_path = interim_dir / "market_sample_ids_0_4.feather"
    label_output_path = interim_dir / "label_sample_ids_0_4.feather"
    market_sample.write_ipc(market_output_path)
    label_sample.write_ipc(label_output_path)

    print(f"selected row count = {selected_row_count}")
    print(f"market sample shape = {market_sample.shape}")
    print(f"label sample shape = {label_sample.shape}")
    print(f"sample IDs = {actual_ids}")
    print(f"market output = {market_output_path}")
    print(f"label output = {label_output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-column", action="store_true")
    parser.add_argument("--market-path", type=Path)
    parser.add_argument("--column-name")
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.extract_column:
        extract_one_column(
            arguments.market_path,
            arguments.column_name,
            arguments.row_count,
            arguments.output_path,
        )
    else:
        current_project_dir = Path(__file__).resolve().parents[1]
        build_sample(current_project_dir)
