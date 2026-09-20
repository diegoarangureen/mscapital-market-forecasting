from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 使用阶段 3 的安全 market 小样本。
# Use the safe market sample from stage 3.
market_sample_path = project_dir / "data" / "interim" / "market_sample_ids_0_4.feather"
market_data = pd.read_feather(market_sample_path)

# 保存增加新列之前的 shape，稍后用它检查行列数量是否符合预期。
# Save the shape before adding a column so the later checks can compare row and column counts.
original_shape = market_data.shape

# TODO 1：判断每行的 transaction_count 是否大于 0，创建布尔 Series。
# TODO 1: Check whether transaction_count is greater than 0 to create a Boolean Series.
has_transactions = market_data["transaction_count"]>0

# 把布尔 Series 增加为新列；不覆盖原始 transaction_count。
# Add the Boolean Series as a new column without replacing transaction_count.
market_data["has_transactions"] = has_transactions

# 使用 notna() 判断每行的平均成交价是否存在。
# Use notna() to check whether the average transaction price exists in each row.
average_price_available = market_data["transaction_avgprice"].notna()

# TODO 2：逐行比较 has_transactions 与 average_price_available，再用 all() 汇总。
# TODO 2: Compare the two Series row by row, then summarize with all().
indicator_matches_every_row = (has_transactions==average_price_available).all()

# 保存处理后的 shape，并分别检查没有丢行、只增加了一列。
# Save the processed shape and separately verify that no rows were lost and one column was added.
processed_shape = market_data.shape
row_count_unchanged = original_shape[0] == processed_shape[0]
one_column_added = processed_shape[1] == original_shape[1] + 1

# 打印结构和一致性检查，作为本课的主要验收结果。
# Print structural and consistency checks as the main lesson validation output.
print(f"original shape = {original_shape}")
print(f"processed shape = {processed_shape}")
print(f"has_transactions dtype = {market_data['has_transactions'].dtype}")
print(f"row count unchanged = {row_count_unchanged}")
print(f"one column added = {one_column_added}")
print(f"indicator matches every row = {indicator_matches_every_row}")
print()

# 统计有成交和无成交各有多少行，确认指示列不是恒定值。
# Count rows with and without transactions to confirm the indicator is not constant.
print("has_transactions counts:")
print(market_data["has_transactions"].value_counts())
print()

# 再次统计原字段缺失数，证明增加指示列没有覆盖或填补原始 NaN。
# Count the original missing values again to prove that adding the indicator did not replace NaNs.
print("original missing values are still present:")
print(market_data["transaction_avgprice"].isna().sum())
