# 第 4 课 A：把一段 market 序列压缩成一行特征

## 本课所处阶段

阶段 4：建立 LightGBM 强基线。

你已经在上一项目掌握 LightGBM、OOF 和 early stopping，所以本阶段不重学树模型。本课解决一个本比赛才出现的新问题：

```text
LightGBM 需要二维表格：一行代表一个训练样本
但是 market 中一个 sample_id 有约 100～200 行时间序列
```

我们必须先把一整段序列汇总成一行。

## 本课目标

完成后，你应该能够：

1. 解释“序列聚合（sequence aggregation）”为什么能把 market 交给表格模型。
2. 使用命名聚合 `groupby().agg()`，为每个 `sample_id` 生成一行特征。
3. 使用 `reset_index()` 把分组键从索引恢复成普通列。
4. 区分总和、均值、比例、最大值和行数分别保留了什么信息。
5. 说明这些聚合为什么暂时没有使用 label，也没有目标泄露。

本课不做这些事情：

- 不重新讲决策树、boosting 或 LightGBM 原理；
- 不训练模型；
- 不读取完整 4.09 GiB market；
- 不加入 order 或 transaction 原始事件流；
- 不一次创造几十个相似特征；
- 不使用 target 选择特征。

## 配套阅读

### 本课不安排新的外部阅读

原因是：

- `groupby().agg()` 和 `reset_index()` 是 pandas 接口，由本讲义直接讲解。
- 序列怎样聚合成表格特征，是当前比赛数据结构带来的实践问题。
- 树模型已在上一项目学过，短期内不重复安排 ISLP 树模型章节。

本课的时间应放在理解“每个统计量丢掉什么、保留什么”，而不是阅读 API 源文档。

## 1. 为什么必须聚合？

label 表中：

```text
sample_id=0 → 一个 target
```

market 表中：

```text
sample_id=0 → 199 行市场过程
```

LightGBM 常见输入形状是：

```text
(样本数, 特征数)
```

所以我们要做：

```text
199 行 market
→ 计算这段过程的统计摘要
→ 1 行 market 特征
→ 与 sample_id=0 的一个 target 对齐
```

例如，一段成交次数：

```text
[0, 2, 3, 0]
```

可以提取：

```text
总成交次数 = 5
有成交的时间段比例 = 2 / 4 = 0.5
时间步数 = 4
```

原来的顺序细节被压缩了，但整体活跃程度被保留下来。这就是表格基线的取舍。

## 2. 聚合统计量分别表达什么？

### `sum`：总量

```text
transaction_volume = [0, 100, 300]
sum = 400
```

适合回答“整个观察窗口一共发生多少”。

### `mean`：平均水平

```text
bid_price_1 = [0.99, 1.00, 1.01]
mean = 1.00
```

适合回答“这段时间通常处于什么水平”。pandas 的 `mean` 默认忽略 `NaN`，所以 `transaction_avgprice_mean` 表示**有成交时间段中的平均价格水平**，不是把无成交段当作 0。

### `max` 和 `min`：范围端点

对 `seconds_before_predict`：

- `max`：这一样本最早覆盖到离预测时点多少秒；
- `min`：最后记录离预测时点还有多少秒。

### `size`：一共有多少行

`size` 统计一个样本实际存在多少个 market 时间步，包括字段中带 `NaN` 的行。

### 布尔均值：比例

pandas 在求布尔均值时，可以把：

```text
True  看作 1
False 看作 0
```

例如：

```text
[True, False, True, True]
```

均值为：

$$
\frac{1 + 0 + 1 + 1}{4} = 0.75
$$

因此 `has_transactions` 的均值就是“有成交的时间段比例”。

## 3. `DataFrameGroupBy.agg()`

### 它解决什么问题

第 3B 课的 `.size()` 只能得到每组行数。`.agg()` 可以让每个分组同时计算多个统计量，并把结果组合成一张特征表。

### 来源与导入

它是 pandas 分组对象的方法。项目已经导入：

```python
import pandas as pd
```

### 分步语法

先分组：

```python
grouped_data = data.groupby("group_column")
```

再命名聚合：

```python
summary = grouped_data.agg(
    output_name=("source_column", "operation"),
)
```

每一项的结构是：

```text
新特征名 = (原始列名, 聚合操作)
```

- 新特征名：输出表中的列名，写在等号左边，不加引号。
- 原始列名：字符串，例如 `"score"`。
- 聚合操作：字符串，例如 `"mean"`、`"sum"` 或 `"size"`。
- 返回：普通 `pd.DataFrame`；分组键默认成为它的索引。

### 脱离比赛的最小例子

```python
students = pd.DataFrame(
    {
        "class_name": ["A", "A", "B"],
        "score": [80, 90, 70],
        "practice_count": [2, 3, 4],
    }
)

grouped_students = students.groupby("class_name")

class_summary = grouped_students.agg(
    student_count=("score", "size"),
    score_mean=("score", "mean"),
    practice_total=("practice_count", "sum"),
)
```

结果类似：

```text
            student_count  score_mean  practice_total
class_name
A                       2        85.0               5
B                       1        70.0               4
```

输入有3名学生，输出有2个班级，所以 shape 从 `(3, 3)` 变为 `(2, 3)`。此时 `class_name` 是索引，没有计入3个特征列。

### 当前比赛中的作用

```python
grouped_market = market_data.groupby("sample_id")
```

再对成交、盘口、时间覆盖和空档状态分别聚合。输入838行，输出应该只有5行，因为小样本中有5个 `sample_id`。

### 常见错误一：左右两边含义写反

错误理解：

```text
原始列 = (新列, 操作)
```

正确理解：

```text
新列 = (原始列, 操作)
```

### 常见错误二：把操作写成圆括号调用

本课命名聚合中传入的是操作名称字符串：

```python
score_mean=("score", "mean")
```

不要写成 `"mean"()`。

## 4. 本课要生成哪些特征？

请按照下表写入 `.agg()`：

| 输出特征名 | 原始列 | 操作 | 表达的信息 |
|---|---|---|---|
| `market_row_count` | `seconds_before_predict` | `size` | 实际时间步数 |
| `seconds_max` | `seconds_before_predict` | `max` | 最早覆盖范围 |
| `seconds_min` | `seconds_before_predict` | `min` | 最接近预测时点的记录 |
| `long_gap_count` | `has_long_gap` | `sum` | 超过4.5秒的空档数 |
| `transaction_volume_sum` | `transaction_volume` | `sum` | 总成交量 |
| `transaction_count_sum` | `transaction_count` | `sum` | 总成交次数 |
| `has_transaction_ratio` | `has_transactions` | `mean` | 有成交时间段比例 |
| `transaction_avgprice_mean` | `transaction_avgprice` | `mean` | 有成交时的平均价格水平 |
| `ask_price_1_mean` | `ask_price_1` | `mean` | 第一档卖价平均水平 |
| `bid_price_1_mean` | `bid_price_1` | `mean` | 第一档买价平均水平 |
| `ask_volume_1_mean` | `ask_volume_1` | `mean` | 第一档卖量平均水平 |
| `bid_volume_1_mean` | `bid_volume_1` | `mean` | 第一档买量平均水平 |

这些只是第一批基线特征。后续会增加波动、买卖价差、买卖不平衡和首尾变化，但本课先保证“一段序列变一行”的结构完全正确。

## 5. `DataFrame.reset_index()`

### 它解决什么问题

`groupby("sample_id")` 聚合后，`sample_id` 默认成为行索引：

```text
           market_row_count  ...
sample_id
0                         199
1                         107
```

为了以后与 label 对齐，我们希望 `sample_id` 回到普通列：

```text
sample_id  market_row_count  ...
0          199
1          107
```

### 来源与基本语法

它是 pandas `DataFrame` 的方法：

```python
data_with_columns = data_with_index.reset_index()
```

- 输入对象：一个 `pd.DataFrame`。
- 本课不传参数。
- 返回：新的 `pd.DataFrame`。
- 原来的 DataFrame 不会自动改变，所以要保存返回值。

### 最小例子

```python
class_summary = class_summary.reset_index()
```

上例会把 `class_name` 从索引恢复为第一列。shape 从 `(2, 3)` 变成 `(2, 4)`：行数不变，多出的普通列就是原分组键。

### 当前比赛中的作用

聚合完成后执行：

```python
sample_features = sample_features.reset_index()
```

预期输出 shape 为 `(5, 13)`：

```text
5行 = 5个sample_id
13列 = sample_id + 12个聚合特征
```

### 常见错误

只写：

```python
sample_features.reset_index()
```

虽然方法产生了新 DataFrame，但没有保存返回值，原变量仍然保持旧索引结构。

## 6. 这会不会产生目标泄露？

本课完全没有读取 label，也没有使用 `target`。

每行特征只来自同一个 `sample_id` 在预测时点之前的 market 数据：

```text
sample 0 的过去 market → sample 0 的聚合特征
sample 1 的过去 market → sample 1 的聚合特征
```

所以当前聚合不会把其他样本的答案带进来。

但以后如果进行标准化、缺失填补或从全体样本学习统计量，仍必须先按月份划分，只在训练月份 `fit()`。这条边界没有改变。

## 7. 你的动手任务

练习文件：

```text
scripts/lesson_04a_aggregate_market_features.py
```

请完成三个 TODO：

1. 按 `sample_id` 创建分组对象。
2. 根据第4节表格，使用命名 `.agg()` 生成12个聚合特征。
3. 使用 `reset_index()` 把 `sample_id` 恢复成普通列。

排序、时间差、长空档和成交指示列都是上一阶段已学内容，固定代码已经提供并加注释。

## 验收标准

- 原 market shape 为 `(838, 13)`。
- 聚合后 shape 为 `(5, 13)`。
- 每个 `sample_id` 恰好占一行。
- 输出包含 `sample_id` 和12个指定特征。
- `market_row_count` 与第3B课的分组计数一致。
- `transaction_avgprice_mean` 可以忽略原始 `NaN` 正常计算。
- 没有读取 label、target 或完整 market 文件。

## 运行后回答

1. 为什么838行market聚合后变成5行？
2. `transaction_count_sum` 和 `has_transaction_ratio` 分别表达什么，为什么不能互相替代？
3. 这种表格聚合丢掉了原始序列中的哪类信息？

## 本课人话总结

LightGBM不会直接阅读一段长度不一的市场电影，所以我们先为每段电影写一张摘要卡：有多少帧、成交多不多、价格通常在哪、有没有时间空档。摘要卡牺牲了逐秒顺序，却让每个 `sample_id` 变成一行，终于具备与一个 `target` 对齐并训练表格模型的结构。
