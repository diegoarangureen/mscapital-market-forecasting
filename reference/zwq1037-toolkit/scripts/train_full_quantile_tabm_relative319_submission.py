"""Train the promoted 319-feature Quantile TabM full-data candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp_tabm_016_relative_scale_features as relative
import train_full_quantile_tabm_b001_submission as full


OUTPUT_STEM = "tabm_quantile_rel319_coslr15_tree053r_blend75_fulltrain"


def add_split_relative_features(project_dir, data: pd.DataFrame, split: str) -> None:
    price = pd.read_feather(
        project_dir
        / "data"
        / "processed"
        / f"{split}_market_microstructure_features.feather",
        columns=["sample_id", *relative.PRICE_RELATIVE_COLUMNS],
    ).sort_values("sample_id")
    if not np.array_equal(data["sample_id"].to_numpy(), price["sample_id"].to_numpy()):
        raise AssertionError(f"{split} relative-price feature IDs are not aligned.")
    for column in relative.PRICE_RELATIVE_COLUMNS:
        data[column] = price[column].to_numpy(dtype=np.float32, copy=False)

    data["x_recent60_trade_count_share"] = relative.safe_ratio(
        data["last60_transaction_count_sum"], data["transaction_count_sum"]
    )
    data["x_recent60_trade_volume_share"] = relative.safe_ratio(
        data["last60_transaction_volume_sum"], data["transaction_volume_sum"]
    )
    data["x_trade_count_peak_share"] = relative.safe_ratio(
        data["transaction_count_max"], data["transaction_count_sum"]
    )
    data["x_trade_volume_peak_share"] = relative.safe_ratio(
        data["transaction_volume_max"], data["transaction_volume_sum"]
    )


def main() -> None:
    base_train_loader = full.tabm.load_exp053r_data
    base_test_loader = full.load_test_only

    def load_train(project_dir):
        data, columns, public_columns, dropped_columns = base_train_loader(project_dir)
        add_split_relative_features(project_dir, data, "train")
        candidate_columns = [*columns, *relative.RELATIVE_COLUMNS]
        if len(candidate_columns) != 319 or len(set(candidate_columns)) != 319:
            raise AssertionError("Expected 319 unique train features.")
        return data, candidate_columns, public_columns, dropped_columns

    def load_test(project_dir):
        data, template, columns = base_test_loader(project_dir)
        add_split_relative_features(project_dir, data, "test")
        candidate_columns = [*columns, *relative.RELATIVE_COLUMNS]
        if len(candidate_columns) != 319 or len(set(candidate_columns)) != 319:
            raise AssertionError("Expected 319 unique test features.")
        return data, template, candidate_columns

    full.OUTPUT_STEM = OUTPUT_STEM
    full.tabm.load_exp053r_data = load_train
    full.load_test_only = load_test
    full.main()


if __name__ == "__main__":
    main()
