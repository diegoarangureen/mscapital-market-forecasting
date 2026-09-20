# 第 1 课 B：第一次读取标签表

## 本课目标

完成后，你应该能回答：

1. 标签表一共有多少个样本和字段？
2. `month`、`sample_id`、`target` 分别扮演什么角色？
3. `target` 是类别标签还是连续数值？
4. 数据覆盖多少个月份，是否有缺失值？
5. 为什么现在可以使用 `pandas.read_feather()`，但不能用同样方式直接读取 4.4 GB 的 market 表？

## 配套阅读

本课不要求阅读 ISLP 或 D2L。它属于数据理解阶段，重点是亲手观察真实文件。

## 1. 三列数据的直觉

标签文件的元数据已经确认包含：

```text
month       时间分组
sample_id   样本身份标识
target      模型需要预测的答案
```

可以先把一行理解为：

```text
在某个月份中，一个编号为 sample_id 的预测任务，其真实答案是 target。
```

后续 market/order/transaction 表必须通过标识信息与这些样本对齐。实际对齐规则要等读取大表结构后确认。

## 2. `pandas.read_feather()`

### 它解决什么问题

把磁盘中的 `.feather` 二进制表格恢复成 pandas `DataFrame`。

### 来源与导入

它是 pandas 模块中的函数，需要：

```python
import pandas as pd
```

### 基本语法

```python
data = pd.read_feather(file_path)
```

### 输入和返回值

- 输入：字符串或 `Path` 路径，指向 Feather 文件。
- 返回：一个 pandas `DataFrame`。
- 返回对象的形状：`(行数, 列数)`，需要运行后实际查看。

### 脱离比赛的最小例子

假设磁盘中有：

```text
students.feather
```

读取方法是：

```python
student_data = pd.read_feather("students.feather")
print(student_data.shape)
```

### 在本比赛中的作用

本课用它读取约 10 MB 的标签表，得到一个包含 `month`、`sample_id`、`target` 的 DataFrame。

### 常见注意事项

`read_feather()` 默认会把所选数据读入内存。10 MB 标签表可以这样做，但 4.4 GB 的 market 文件解码后占用可能远大于 4.4 GB，不能直接照搬。

## 3. `Series.nunique()`

### 它解决什么问题

计算一列中有多少个不同的值。

### 基本语法

```python
unique_count = data["column_name"].nunique()
```

### 输入和返回值

- 调用者：一个 pandas `Series`，也就是 DataFrame 中的一列。
- 默认忽略缺失值。
- 返回：整数。

### 最小例子

```python
colors = pd.Series(["red", "red", "blue"])
color_count = colors.nunique()
print(color_count)  # 2
```

### 当前作用

- `month.nunique()`：数据中出现多少个不同月份编号。
- `sample_id.nunique()`：不同样本编号数量是否与总行数一致。

### 常见错误

`nunique()` 是方法，需要括号。`data["month"].nunique` 只会得到方法对象，不会执行统计。

## 4. `Series.describe()`

### 它解决什么问题

一次生成数值列的常用描述统计。

### 基本语法

```python
summary = data["column_name"].describe()
```

### 返回值

返回 pandas `Series`，通常含有：

```text
count, mean, std, min, 25%, 50%, 75%, max
```

### 最小例子

```python
scores = pd.Series([60, 70, 80])
score_summary = scores.describe()
print(score_summary)
```

### 当前作用

观察 `target` 的范围、均值和波动，初步判断它是不是连续预测目标。

### 注意事项

`describe()` 只是在描述已有数据，不代表目标服从正态分布，也不说明预测一定容易。

## 5. 你的动手任务

骨架文件已经准备：

```text
scripts/lesson_01b_inspect_label.py
```

请完成其中的 TODO，打印：

1. DataFrame 的类型、`shape`、列名和每列类型。
2. 前 5 行。
3. 每列缺失值数量。
4. `month` 的最小值、最大值和不同值数量。
5. `sample_id` 的不同值数量。
6. `target.describe()` 的结果。

不要删除或修改原始 Feather 文件。

## 验收标准

- 脚本能由项目 Conda 环境运行。
- 输出包含上述六类信息。
- 学习者能用自己的话回答：一行表示什么、哪列不能放入模型特征、为什么 market 表不能直接完整读入 pandas。

