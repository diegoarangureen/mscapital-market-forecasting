from pathlib import Path

import pandas as pd


# 找到当前比赛项目目录。
# Locate the current competition project directory.
project_dir = Path(__file__).resolve().parents[1]

# 指向保持不变的原始标签文件。
# Point to the immutable raw label file.
label_path = project_dir / "data" / "raw" / "label.feather"

# TODO 1：使用 pd.read_feather() 读取标签表。
# TODO 1: Load the label table with pd.read_feather().
data=pd.read_feather(label_path)
# TODO 2：打印对象类型、shape、列名和每列的数据类型。
# TODO 2: Print the object type, shape, column names, and column data types.
print(f"data: {type(data),data.shape,data.columns,data.dtypes}")
# TODO 3：打印前 5 行和每列缺失值数量。
# TODO 3: Print the first 5 rows and missing-value count of each column.
print(data[:5])
print(data.isna().sum())
# TODO 4：打印 month 的最小值、最大值和不同值数量。
# TODO 4: Print the minimum, maximum, and unique-value count of month.
print(data["month"].min())
print(data["month"].max())
print(data["month"].nunique())
# TODO 5：打印 sample_id 的不同值数量。
# TODO 5: Print the unique-value count of sample_id.
print(data["sample_id"].nunique())
# TODO 6：打印 target 的 describe() 结果。
# TODO 6: Print the describe() result of target.
print(data["target"].describe())
