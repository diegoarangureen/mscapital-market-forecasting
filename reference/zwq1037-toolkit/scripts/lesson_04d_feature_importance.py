from pathlib import Path

import pandas as pd
from lightgbm import Booster
from pandas.core.interchange.dataframe_protocol import DataFrame

# 找到上一课保存的最佳轮数模型。
# Locate the best-iteration model saved in the previous lesson.
project_dir = Path(__file__).resolve().parents[1]
model_path = project_dir / "outputs" / "models" / "lightgbm_market_v1.txt"

# 先由Python读取模型文本，避免LightGBM文件接口处理中文路径时失败。
# Let Python read the model text first to avoid Unicode-path problems in LightGBM's file interface.
model_text = model_path.read_text(encoding="utf-8")
model = Booster(model_str=model_text)

# 名称与两种重要性数组使用相同顺序，长度都应为40。
# Names and both importance arrays share the same order and should all have length 40.
feature_names = model.feature_name()
gain_values = model.feature_importance(importance_type="gain")
split_values = model.feature_importance(importance_type="split")

print(f"feature count = {len(feature_names)}")
print(f"gain shape = {gain_values.shape}")
print(f"split shape = {split_values.shape}")

# TODO 1：建立三列DataFrame：feature、gain、split_count。
# TODO 1: Build a three-column DataFrame: feature, gain, and split_count.
importance_table = pd.DataFrame(
{
    "feature": feature_names,
    "gain": gain_values,
    "split_count": split_values,
}
)
# TODO 2A：把gain总和保存到total_gain。
# TODO 2A: Store the sum of gain in total_gain.
total_gain = importance_table["gain"].sum()

# TODO 2B：新增gain_percentage列：每行gain除以总gain，再乘100。
# TODO 2B: Add gain_percentage: each row's gain divided by total gain, times 100.
importance_table["gain_percentage"] = importance_table["gain"]/total_gain*100

# TODO 3：按gain_percentage降序排序，并保存为sorted_importance_table。
# TODO 3: Sort by gain_percentage in descending order and save the result.
sorted_importance_table = importance_table.sort_values(
    by=["gain_percentage"], ascending=False
)

# 固定验收：表应有40行4列，百分比之和约为100，首行不小于末行。
# Fixed checks: expect 40 rows and 4 columns, percentages near 100, and descending endpoints.
assert sorted_importance_table.shape == (40, 4)
assert abs(sorted_importance_table["gain_percentage"].sum() - 100.0) < 1e-6
assert (
    sorted_importance_table.iloc[0]["gain_percentage"]
    >= sorted_importance_table.iloc[-1]["gain_percentage"]
)

print()
print("Top 10 features by gain:")
print(
    sorted_importance_table[
        ["feature", "gain_percentage", "split_count"]
    ].head(10).to_string(index=False)
)
print()
print(f"importance table shape = {sorted_importance_table.shape}")
print(
    "gain percentage sum = "
    f"{sorted_importance_table['gain_percentage'].sum():.6f}"
)

# 运行后用中文写下你的观察；这是本课需要亲自完成的结果解释。
# After running, write your observations in Chinese; this interpretation is the core learning task.
# 1. gain 排名前三的特征是什么？ask_price_1_last：最后记录的第一档卖价
# bid_price_1_last：最后记录的第一档买价
# spread_1_mean：整段序列的平均买卖价差
# 2. split 次数最多的特征是否一定也是 gain 最大的特征？根据输出回答。不一定是，但似乎有某种相关
# 3. 为什么这张表只能帮助我们提出下一步假设，不能证明因果关系？你的问题不该这样问，你应该问这张表能够帮助我做什么？同时阐述下一步我们可能做什么，让我来做选择，然后你评价
