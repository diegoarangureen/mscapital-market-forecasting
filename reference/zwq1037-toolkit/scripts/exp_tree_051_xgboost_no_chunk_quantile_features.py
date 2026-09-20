"""Remove public features that depend on cross-sample chunk quantiles from EXP050."""

from exp_tree_050_xgboost_rfmf_deduplicated import run_experiment


CHUNK_QUANTILE_FEATURES = (
    "t_large_buy_90",
    "t_large_sell_95",
    "x_large_trade_imbalance",
)


if __name__ == "__main__":
    run_experiment(
        experiment_id="EXP-TREE-051",
        extra_excluded_public_features=CHUNK_QUANTILE_FEATURES,
        description=(
            "EXP050 without three public features that depend on cross-sample "
            "40k-ID chunk volume quantiles"
        ),
    )
