"""Run the exact book builder using the globally audited raw row order."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import build_exact_market_book_features as exact_builder
from build_market_aggregate_features import safe_write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    arguments = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]

    original_sort = pl.LazyFrame.sort

    def skip_verified_source_sort(self, by, *args, **kwargs):
        if by == ["sample_id", "seconds_before_predict"]:
            return self
        return original_sort(self, by, *args, **kwargs)

    pl.LazyFrame.sort = skip_verified_source_sort
    try:
        output = exact_builder.build_features(project_dir, arguments.split)
    finally:
        pl.LazyFrame.sort = original_sort

    output_path = (
        project_dir
        / "data"
        / "processed"
        / f"{arguments.split}_market_book_features_exact.feather"
    )
    safe_write(output, output_path)
    print(f"shape = {output.shape}", flush=True)
    print(f"output = {output_path}", flush=True)


if __name__ == "__main__":
    main()
