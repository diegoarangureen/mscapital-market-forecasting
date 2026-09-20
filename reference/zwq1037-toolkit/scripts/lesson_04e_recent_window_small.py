from pathlib import Path

import pandas as pd
from torch.fx.experimental.unification.unification_tools import groupby

# 找到阶段3已经安全保存的5个完整market样本。
# Locate the five complete market samples safely saved in stage three.
project_dir = Path(__file__).resolve().parents[1]
sample_path = (
    project_dir
    / "data"
    / "interim"
    / "market_sample_ids_0_4.feather"
)

# 读取小样本；本课不直接触碰2.2亿行的完整market文件。
# Read the small sample; this lesson does not directly touch the 221.7-million-row market file.
market_sample = pd.read_feather(sample_path)

print(f"original shape = {market_sample.shape}")
print(
    "original seconds range = "
    f"{market_sample['seconds_before_predict'].min():.3f} to "
    f"{market_sample['seconds_before_predict'].max():.3f}"
)

# TODO 1：逐行判断倒计时是否位于最后60秒，得到布尔Series。
# TODO 1: Test whether each countdown is in the final 60 seconds to obtain a Boolean Series.
recent_condition = market_sample["seconds_before_predict"] <= 60
# TODO 2：使用recent_condition筛选所有列，并创建独立的小窗口表。
# TODO 2: Use recent_condition to select all columns and create an independent window table.
recent_market = market_sample[recent_condition].copy()

# TODO 3：新增逐行第一档价差：卖一价减去买一价。
# TODO 3: Add the row-level level-one spread: best ask minus best bid.
recent_market["spread_1"] = (recent_market.ask_price_1 - recent_market.bid_price_1)

# TODO 4：按sample_id聚合，生成下列四个窗口特征，然后reset_index()：
# recent_row_count、ask_price_1_mean_last60、ask_price_1_mean_last60、spread_1_mean_last60。
# TODO 4: Aggregate by sample_id into the four window features below, then call reset_index().
recent_features = (
    recent_market.groupby("sample_id")
    .agg(
        recent_row_count=("seconds_before_predict", "size"),
        ask_price_1_mean_last60=("ask_price_1", "mean"),
        bid_price_1_mean_last60=("bid_price_1", "mean"),
        spread_1_mean_last60=("spread_1", "mean"),
    )
    .reset_index()
)

# 固定验收：窗口方向、样本覆盖和输出shape必须正确。
# Fixed checks: the window direction, sample coverage, and output shape must be correct.
assert recent_market["seconds_before_predict"].max() <= 60
assert recent_market["sample_id"].nunique() == 5
assert recent_features.shape == (5, 5)
assert recent_features["sample_id"].is_unique

print()
print(f"recent market shape = {recent_market.shape}")
print(
    "recent seconds range = "
    f"{recent_market['seconds_before_predict'].min():.3f} to "
    f"{recent_market['seconds_before_predict'].max():.3f}"
)
print("recent rows per sample:")
print(
    recent_features[
        ["sample_id", "recent_row_count"]
    ].to_string(index=False)
)
print()
print(recent_features.to_string(index=False))

# 运行前判断：C是否大概率不成立？在A和B中你更倾向哪一个？为什么？
# Pre-run judgment: Is C likely invalid? Which of A and B do you lean toward, and why?
# 你的判断：
'''我其实觉得B可能会好？
最后60s特征其实之前也有包含部分，现在单独拿出来提升不一定特别大'''
# 运行后回答：为什么最后60秒使用 <= 60，而不是 >= 60？
# Post-run answer: Why does the final 60-second window use <= 60 rather than >= 60?
# 你的回答：seconds_before是距离交易的时间
