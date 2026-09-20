from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 使用阶段 3 中已经验证过的 5 个完整 market 样本。
# Use the five complete market samples validated in stage 3.
market_sample_path = project_dir / "data" / "interim" / "market_sample_ids_0_4.feather"
market_data = pd.read_feather(market_sample_path)

# 保存原始小样本的 shape，区分原始 13 列和后面临时增加的分析列。
# Save the raw sample shape to distinguish its 13 columns from later temporary analysis columns.
original_market_shape = market_data.shape

# 按样本编号和倒计时排序，保证相邻差值只描述正确的时间顺序。
# Sort by sample ID and countdown so adjacent differences follow the correct time order.
market_data = market_data.sort_values(
    by=["sample_id", "seconds_before_predict"],
    ascending=[True, False],
).copy()

# 复用第 3C 课的逻辑，计算每个样本内部的正数时间间隔。
# Reuse lesson 3C logic to compute positive time gaps within each sample.
grouped_seconds = market_data.groupby("sample_id")["seconds_before_predict"]
countdown_changes = grouped_seconds.diff()
market_data["time_gap_seconds"] = -countdown_changes

# 标记明显长于典型约 3 秒间隔的位置；4.5 只是当前观察阈值。
# Mark gaps clearly longer than the typical three seconds; 4.5 is only an inspection threshold.
market_data["has_long_gap"] = market_data["time_gap_seconds"] > 4.5

# 复用第 3D 课的规则，显式记录当前时间段是否发生成交。
# Reuse lesson 3D logic to record whether transactions occurred in each time period.
market_data["has_transactions"] = market_data["transaction_count"] > 0

print(f"original market shape = {original_market_shape}")
print(f"working market shape before aggregation = {market_data.shape}")
print()

# TODO 1：按照 sample_id 创建 DataFrame 分组对象。
# TODO 1: Create a DataFrame group object by sample_id.
grouped_market = market_data.groupby("sample_id")

# TODO 2：使用讲义第 4 节的表格，命名聚合出 12 个特征。
# TODO 2: Use the table in lesson section 4 to create 12 named aggregate features.
sample_features = grouped_market.agg(
    market_row_count=("seconds_before_predict", "size"),
    seconds_max=("seconds_before_predict", "max"),
    seconds_min=("seconds_before_predict", "min"),
    long_gap_count=("has_long_gap", "sum"),
    transaction_volume_sum=("transaction_volume", "sum"),
    transaction_count_sum=("transaction_count", "sum"),
    has_transaction_ratio=("has_transactions", "mean"),
    transaction_avgprice_mean=("transaction_avgprice", "mean"),
    ask_price_1_mean=("ask_price_1", "mean"),
    bid_price_1_mean=("bid_price_1", "mean"),
    ask_volume_1_mean=("ask_volume_1", "mean"),
    bid_volume_1_mean=("bid_volume_1", "mean"),
)
# TODO 3：把 sample_id 从索引恢复成普通列，并保存返回的新 DataFrame。
# TODO 3: Restore sample_id from the index to a regular column and save the returned DataFrame.
sample_features = sample_features.reset_index()

# 打印输出结构，确认多行序列已经变成每个 sample_id 一行。
# Print the output structure to confirm that each multi-row sequence became one row per sample ID.
print(f"feature table type = {type(sample_features)}")
print(f"feature table shape = {sample_features.shape}")
print(f"feature columns = {sample_features.columns.tolist()}")
print()

# 打印完整的 5 行小表，方便逐样本核对行数、空档和成交活跃度。
# Print all five rows so row counts, gaps, and transaction activity can be checked per sample.
print(sample_features.to_string(index=False))
