from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 完整 market 特征已经由逐列、短进程生成器安全构建。
# Full market features were safely built by the column-wise short-process generator.
feature_path = project_dir / "data" / "processed" / "train_market_features.feather"
label_path = project_dir / "data" / "raw" / "label.feather"

# 读取已经压缩到每个 sample_id 一行的特征表和标签表。
# Read feature and label tables that each contain one row per sample ID.
feature_data = pd.read_feather(feature_path)
label_data = pd.read_feather(label_path)

# TODO 1：按 sample_id 做 inner merge，并要求左右两边都是 one_to_one。
# TODO 1: Inner-merge on sample_id and require a one-to-one relationship on both sides.
model_data = feature_data.merge(
    label_data,
    on="sample_id",
    how="inner",
    validate="one_to_one",
)

# 特征文件第一列是 sample_id；[1:] 取得后面的 40 个模型特征名。
# The first feature-file column is sample_id; [1:] selects the following 40 model feature names.
feature_columns = feature_data.columns[1:].tolist()

# X 只使用 market 特征；y 是一维 target；month 只用于时间划分。
# X contains only market features; y is the target Series; month is used only for time splitting.
X = model_data[feature_columns]
y = model_data["target"]
month = model_data["month"]

# TODO 2：创建训练条件，选择月份 0～59。
# TODO 2: Create the training condition for months 0 through 59.
train_condition = model_data["month"] <= 59

# TODO 3：创建验证条件，选择月份 60～70。
# TODO 3: Create the validation condition for months 60 through 70.
valid_condition = model_data["month"] >= 60

# 使用布尔条件切分；copy() 让四个结果成为独立的训练/验证对象。
# Split with Boolean conditions; copy() makes the four outputs independent train/validation objects.
X_train = X.loc[train_condition].copy()
X_valid = X.loc[valid_condition].copy()
y_train = y.loc[train_condition].copy()
y_valid = y.loc[valid_condition].copy()

# 保存对应月份，只用于验证时间顺序，不作为模型输入。
# Keep corresponding months only to verify temporal order, not as model inputs.
train_month = month.loc[train_condition]
valid_month = month.loc[valid_condition]

# 分开比较三个行数，避免用紧凑的连续比较隐藏检查步骤。
# Compare the three row counts separately instead of hiding checks in a compact chained comparison.
feature_and_label_rows_match = feature_data.shape[0] == label_data.shape[0]
merged_and_feature_rows_match = model_data.shape[0] == feature_data.shape[0]
all_rows_matched = feature_and_label_rows_match and merged_and_feature_rows_match

# 分别检查三个禁止列，避免引入本课不需要学习的生成器表达式。
# Check each forbidden column explicitly to avoid introducing an unnecessary generator expression.
sample_id_absent = "sample_id" not in feature_columns
month_absent = "month" not in feature_columns
target_absent = "target" not in feature_columns
forbidden_columns_absent = sample_id_absent and month_absent and target_absent

# 打印完整表结构和防泄露检查，作为进入模型训练前的验收证据。
# Print full-table structure and leakage checks as evidence before model training.
print(f"feature shape = {feature_data.shape}")
print(f"label shape = {label_data.shape}")
print(f"merged shape = {model_data.shape}")
print(f"all rows matched = {all_rows_matched}")
print(f"model feature count = {len(feature_columns)}")
print(f"forbidden columns absent = {forbidden_columns_absent}")
print(f"feature missing values = {int(X.isna().sum().sum())}")
print()

# 打印训练和验证 shape，以及两部分的月份边界。
# Print train/validation shapes and the month boundaries of both partitions.
print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")
print(f"train months = {train_month.min()} to {train_month.max()}")
print(f"X_valid shape = {X_valid.shape}")
print(f"y_valid shape = {y_valid.shape}")
print(f"valid months = {valid_month.min()} to {valid_month.max()}")
print(f"time order is valid = {train_month.max() < valid_month.min()}")
