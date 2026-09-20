from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 继续使用已经安全导出的 5 个完整样本。
# Continue using the five safely exported complete samples.
market_sample_path = project_dir / "data" / "interim" / "market_sample_ids_0_4.feather"
market_data = pd.read_feather(market_sample_path)

# 先按样本编号排列，再让每个样本内部从较早时刻走向预测时点。
# Sort by sample ID, then move from earlier observations toward prediction time.
market_data = market_data.sort_values(
    by=["sample_id", "seconds_before_predict"],
    ascending=[True, False],
).copy()

print(f"market shape = {market_data.shape}")
print()

# 每个样本只能和自己内部的上一时刻比较。
# Each sample may only be compared with its own previous observation.
grouped_seconds = market_data.groupby("sample_id")["seconds_before_predict"]

# TODO 1：对 grouped_seconds 调用 diff()，计算“当前倒计时减去上一行倒计时”。
# TODO 1: Call diff() on grouped_seconds to compute current minus previous countdown.
countdown_changes = grouped_seconds.diff()

# 倒计时按从大到小排列，因此加负号把间隔长度变成正数。
# The countdown is descending, so negate the changes to obtain positive gaps.
market_data["time_gap_seconds"] = -countdown_changes

print("time-gap summary:")
for selected_sample_id in [0, 1, 2, 3, 4]:
    selected_sample_condition = market_data["sample_id"] == selected_sample_id
    one_sample_market = market_data.loc[selected_sample_condition]
    one_sample_gaps = one_sample_market["time_gap_seconds"]
    long_gap_count = (one_sample_gaps > 4.5).sum()

    print(
        f"sample {selected_sample_id}: "
        f"rows={one_sample_market.shape[0]}, "
        f"median_gap={one_sample_gaps.median():.3f}, "
        f"max_gap={one_sample_gaps.max():.3f}, "
        f"gaps_over_4.5={long_gap_count}"
    )

print()

# 建立两个需要比较的布尔 Series。
# Build the two Boolean Series to compare.
average_price_missing = market_data["transaction_avgprice"].isna()
no_transactions = market_data["transaction_count"] == 0

# TODO 2：用 == 逐行比较两个布尔 Series。
# TODO 2: Compare the two Boolean Series row by row with ==.
row_by_row_matches = average_price_missing == no_transactions

# TODO 3：用 all() 检查 row_by_row_matches 是否全部为 True。
# TODO 3: Use all() to check whether every value in row_by_row_matches is True.
all_rows_match = row_by_row_matches.all()

print(f"missing average-price rows = {average_price_missing.sum()}")
print(f"zero-transaction rows = {no_transactions.sum()}")
print(f"patterns match on every row = {all_rows_match}")
