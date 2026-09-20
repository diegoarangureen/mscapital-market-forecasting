from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 读取体积较小的标签表。
# Load the relatively small label table.
label_path = project_dir / "data" / "raw" / "label.feather"
label_data = pd.read_feather(label_path)

# 固定第一个时间验证边界。
# Fix the first chronological validation boundary.
train_end_month = 59
valid_start_month = 60

# TODO 1：选择 month 小于或等于 59 的训练行，生成布尔掩码。
# TODO 1: Build a Boolean mask for training rows whose month is at most 59.
train_mask = label_data["month"] <= train_end_month

# TODO 2：选择 month 大于或等于 60 的验证行，生成布尔掩码。
# TODO 2: Build a Boolean mask for validation rows whose month is at least 60.
valid_mask = label_data["month"] >= valid_start_month

# TODO 3：使用 .loc[] 和 .copy() 建立训练数据。
# TODO 3: Create the training data with .loc[] and .copy().
train_data = label_data[train_mask].copy()

# TODO 4：使用 .loc[] 和 .copy() 建立验证数据。
# TODO 4: Create the validation data with .loc[] and .copy().
valid_data = label_data[valid_mask].copy()

# 以下打印代码用于验收，不需要修改。
# The following output code is provided for validation and requires no changes.
print(f"all shape = {label_data.shape}")
print(f"train shape = {train_data.shape}")
print(f"valid shape = {valid_data.shape}")
print()
print(f"train month range = {train_data['month'].min()} to {train_data['month'].max()}")
print(f"valid month range = {valid_data['month'].min()} to {valid_data['month'].max()}")
print()
print(f"all rows = {label_data.shape[0]}")
print(f"train rows + valid rows = {train_data.shape[0] + valid_data.shape[0]}")
print(f"train target mean = {train_data['target'].mean():.6f}")
print(f"valid target mean = {valid_data['target'].mean():.6f}")
