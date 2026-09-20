from pathlib import Path

import polars as pl

from scripts.build_train_market_last60_features import (
    SOURCE_COLUMNS,
    aggregate_one_column,
    combine_parts,
)


def test_last60_builder_on_five_samples(tmp_path: Path) -> None:
    """The full builder must preserve five sample IDs and create 22 features."""

    project_dir = Path(__file__).resolve().parents[1]
    sample_path = (
        project_dir
        / "data"
        / "interim"
        / "market_sample_ids_0_4.feather"
    )

    part_paths = []
    for column_name in SOURCE_COLUMNS:
        part_path = tmp_path / f"{column_name}.feather"
        aggregate_one_column(sample_path, column_name, part_path)
        part_paths.append(part_path)

    output_path = tmp_path / "last60_features.feather"
    combine_parts(part_paths, output_path)
    result = pl.read_ipc(output_path)

    assert result.shape == (5, 23)
    assert result["sample_id"].n_unique() == 5
    assert result["last60_row_count"].to_list() == [20, 16, 11, 20, 21]
    assert "last60_spread_1_mean" in result.columns
    assert "last60_mid_price_1_change" in result.columns
    assert "last60_book_volume_imbalance_1" in result.columns
