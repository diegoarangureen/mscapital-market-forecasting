from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 使用已经安全导出的 5 个完整样本，不读取 4.09 GiB 原始 market 文件。
# Use five safely exported complete samples instead of the 4.09 GiB raw market file.
market_sample_path = project_dir / "data" / "interim" / "market_sample_ids_0_4.feather"
label_sample_path = project_dir / "data" / "interim" / "label_sample_ids_0_4.feather"

# 读取小型 market 和 label 表；这部分是准备工作，不需要修改。
# Read the small market and label tables; this setup code needs no changes.
market_data = pd.read_feather(market_sample_path)
label_data = pd.read_feather(label_sample_path)

print(f"market shape = {market_data.shape}")
print(f"label shape = {label_data.shape}")
print()

# TODO 1：按照 sample_id 分组，并用 size() 统计每组的行数。
# TODO 1: Group by sample_id and count the rows in every group with size().
row_counts=market_data.groupby("sample_id").size()

print("rows in each sample:")
print(row_counts)
print()

# 本课固定观察样本 0。
# Inspect sample 0 in this lesson.
selected_sample_id = 0

# TODO 2：创建布尔条件，判断 market_data 的 sample_id 是否等于 selected_sample_id。
# TODO 2: Create a Boolean condition for sample_id equal to selected_sample_id.
selected_sample_condition = market_data["sample_id"] == selected_sample_id

# 使用上一课已经学过的 .loc[] 和 .copy() 取得独立的小表。
# Use the familiar .loc[] and .copy() to create an independent small table.
one_sample_market = market_data.loc[selected_sample_condition].copy()

# TODO 3：按 seconds_before_predict 从大到小排列 one_sample_market。
# TODO 3: Sort one_sample_market by seconds_before_predict from largest to smallest.
one_sample_market = one_sample_market.sort_values(
    by="seconds_before_predict",
    ascending=False,
)

# 标签筛选是非核心准备代码，本课不要求重写。
# Label filtering is setup code rather than the core exercise.
selected_label_condition = label_data["sample_id"] == selected_sample_id
one_sample_label = label_data.loc[selected_label_condition].copy()

# 固定观察列，避免一次打印整张宽表。
# Use fixed columns to avoid printing the entire wide table.
display_columns = [
    "sample_id",
    "seconds_before_predict",
    "transaction_avgprice",
    "ask_price_1",
    "bid_price_1",
]

print(f"selected sample id = {selected_sample_id}")
print(f"selected market shape = {one_sample_market.shape}")
print(
    "seconds range = "
    f"{one_sample_market['seconds_before_predict'].max():.3f} to "
    f"{one_sample_market['seconds_before_predict'].min():.3f}"
)
print()

print("earliest three displayed rows:")
print(one_sample_market[display_columns].head(3).to_string(index=False))
print()

print("latest three displayed rows:")
print(one_sample_market[display_columns].tail(3).to_string(index=False))
print()

print("missing values in the selected market sample:")
print(one_sample_market.isna().sum())
print()

print("matching label row:")
print(one_sample_label.to_string(index=False))
