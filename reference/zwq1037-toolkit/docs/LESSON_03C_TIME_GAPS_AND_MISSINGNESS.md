# 第 3 课 C：时间空档与“有含义的缺失”

## 本课所处阶段

阶段 3：安全读取并对齐 market 小样本。

第 3B 课发现：5 个 `sample_id` 的行数并不完全相同，而且 `transaction_avgprice` 中存在 `NaN`。本课不急着修补数据，而是先用证据回答：

```text
行数不同，是不是因为某些相邻时间点隔得更远？
平均成交价缺失，是不是与没有成交有关？
```

## 本课目标

完成后，你应该能够：

1. 使用 `.diff()` 计算一条序列中相邻两行的变化。
2. 区分原始数据中的缺失和计算相邻差值时自然产生的缺失。
3. 用两个布尔条件逐行比较，再用 `.all()` 检查假设是否对所有行成立。
4. 解释为什么缺失值可能携带信息，不能看到 `NaN` 就机械填 0 或删行。

本课不做这些事情：

- 不正式填补缺失值；
- 不删除任何 market 行；
- 不建立模型特征；
- 不根据 5 个样本就断言完整数据一定具有相同规律；
- 不读取 4.09 GiB 的原始 market 文件。

## 配套阅读

### 课前选读：ISLP 中关于缺失数据的一小段

- 书名：*An Introduction to Statistical Learning with Applications in Python*，2023 年版。
- 位置：第 2 章 “Statistical Learning”，第 2.3.7 节 “Loading Data”，书本页码第 56 页；在本地 PDF 中是第 66 页。
- 范围：只读从 “There are various ways to deal with missing data” 开始，到 `Auto.dropna()` 示例结束的一小段。
- 时机与深度：写代码前读 5 分钟，只理解作者为什么在那个汽车数据例子中删除 5 行。
- 暂时跳过：`read_csv()`、`na_values`、`dropna()` 的语法，以及本节后面的行列选择代码。这些 API 不是本课阅读任务。

本地教材：[ISLP_2023.pdf](../references/books/ISLP_2023.pdf)

读后思考：

1. 书中的数据有 397 行，只有 5 行缺失；作者选择删除它们。这个决定是否能不加判断地照搬到我们的高频序列？
2. 如果 `transaction_avgprice` 缺失恰好表示“这一时间段没有成交”，删除这行会同时丢掉什么信息？

### 为什么没有安排更多阅读

`.diff()` 和 `.all()` 属于 pandas 接口，由 Codex 查证后在本讲义中完整讲解，不要求阅读源文档。时间间隔和成交字段之间的关系是本比赛的具体数据问题，需要依靠实际检查，而不是再读一章泛化教材。

## 1. 先别急着“处理”缺失值

看到 `NaN` 时，初学者很容易马上问：

```text
应该填平均数、中位数，还是直接删掉？
```

但更早的问题应该是：

```text
它为什么缺失？
```

例如，三分钟内没有下雨时，“平均雨滴大小”可能没有定义。这不代表传感器损坏，而是“没有雨”导致这个统计量无法计算。

market 中的成交字段也可能类似：

- `transaction_count > 0`：这段时间发生过成交，可以计算平均成交价；
- `transaction_count == 0`：没有成交，“平均成交价”可能没有定义，于是成为 `NaN`。

这目前只是**假设**。本课要用数据检验它，而不是把猜测写成事实。

## 2. `Series.diff()`：计算相邻变化

### 它解决什么问题

假设温度按时间排列：

```text
20, 23, 22
```

相邻变化分别是：

```text
第一项：没有上一项，因此无法计算
第二项：23 - 20 = 3
第三项：22 - 23 = -1
```

`.diff()` 自动完成“当前值减去前一个值”。

### 来源与导入

它是 pandas `Series` 的方法。项目已经导入：

```python
import pandas as pd
```

### 基本语法

```python
changes = values.diff()
```

- `values`：一维 `pd.Series`，shape 为 `(行数,)`。
- 默认比较前 1 行，相当于 `values.diff(periods=1)`。
- 返回：新的 `pd.Series`，shape 与输入相同。
- 返回结果的第一项是 `NaN`，因为第一项没有“前一行”可以相减。

### 最小例子

```python
temperatures = pd.Series([20, 23, 22])
temperature_changes = temperatures.diff()
print(temperature_changes)
```

结果：

```text
0    NaN
1    3.0
2   -1.0
```

这里第一项的 `NaN` 是计算产生的，不代表原始温度缺失。

### 当前比赛中的具体作用

我们的 `seconds_before_predict` 已经按从大到小排列，例如：

```text
596, 593, 590
```

直接 `.diff()` 得到：

```text
NaN, -3, -3
```

负号只是因为倒计时正在减小。为了把“间隔长度”写成正数，可以在结果前加一元负号：

```python
time_gaps = -countdown_changes
```

于是得到：

```text
NaN, 3, 3
```

### 为什么要先 `groupby()` 再 `.diff()`？

如果把样本 0 的最后一行和样本 1 的第一行直接相减，就会制造一个跨样本的假间隔。正确关系是每个 `sample_id` 只和自己内部的上一行比较：

```python
grouped_seconds = market_data.groupby("sample_id")["seconds_before_predict"]
countdown_changes = grouped_seconds.diff()
```

`grouped_seconds` 是按 `sample_id` 分组后，只保留倒计时这一列的分组对象。`.diff()` 会在每一组内部重新开始，所以每个样本的第一行都会产生一个 `NaN`。

### 常见错误

`.diff()` 依赖当前行顺序。没有先按 `sample_id` 和时间排序，就无法保证“上一行”真的是上一时刻。

## 3. 中位数时间间隔是什么意思？

本课固定代码会打印每个样本的：

- 行数；
- 相邻间隔的中位数（median）；
- 最大间隔；
- 大于 4.5 秒的间隔数量。

数字例子：

```text
间隔 = [3.0, 3.1, 3.0, 12.0]
```

排序后是 `[3.0, 3.0, 3.1, 12.0]`，中间两个数的平均值为：

$$
\frac{3.0 + 3.1}{2} = 3.05
$$

虽然有一次 12 秒的大空档，中位数仍能告诉我们“典型间隔约为 3 秒”。

本课把 `4.5` 秒作为观察用阈值，因为它比典型的约 3 秒明显更长。它只是帮助寻找可疑空档，不是比赛官方规定，也不是将来模型中的固定超参数。

## 4. 逐行比较两个布尔条件

我们会建立两个布尔 Series：

```python
average_price_missing = market_data["transaction_avgprice"].isna()
no_transactions = market_data["transaction_count"] == 0
```

二者 shape 都是 `(838,)`，每行各有一个 `True` 或 `False`。

使用 `==` 比较两个 Series 时，pandas 会逐行比较：

```python
row_by_row_matches = average_price_missing == no_transactions
```

小数字例子：

```text
average_price_missing = [True,  False, True]
no_transactions       = [True,  False, False]
逐行是否一致          = [True,  True,  False]
```

这说明前两行符合假设，第三行不符合。

## 5. `Series.all()`：是否全部为 True

### 它解决什么问题

如果布尔 Series 很长，不能靠肉眼寻找有没有一个 `False`。`.all()` 回答：所有元素是否全部为 `True`？

### 来源与调用

它是 pandas `Series` 的方法，不需要额外导入：

```python
all_rows_match = row_by_row_matches.all()
```

- 输入对象：布尔 `pd.Series`。
- 参数：本课不传参数。
- 返回：单个布尔值，`True` 或 `False`，不再是 Series。

### 最小例子

```python
first_check = pd.Series([True, True, True])
second_check = pd.Series([True, False, True])

print(first_check.all())   # True
print(second_check.all())  # False
```

### 当前比赛中的具体作用

```python
all_rows_match = row_by_row_matches.all()
```

如果结果为 `True`，只能说明当前 838 行小样本里两个条件完全一致。这是支持假设的强证据，但还不能替代对完整训练数据的检查。

### 常见错误

不要写成：

```python
if row_by_row_matches:
```

一个 Series 中有许多真假值，Python 不知道你想检查“全部为真”还是“至少一个为真”。检查全部为真应明确调用 `.all()`。

## 6. 缺失为什么可能是一种信息？

如果“平均成交价缺失”等价于“没有成交”，那么缺失本身描述了市场状态：

```text
有成交 ↔ 市场在这一小段时间里发生了交易
无成交 ↔ 市场在这一小段时间里没有交易
```

此时直接删除缺失行会同时删除“没有成交的时间段”；直接填 0 又会制造一个远离正常价格尺度的假价格。

以后可能考虑：

- 保留 `transaction_count` 作为“是否成交”的信息；
- 添加缺失指示列；
- 用合理方式填价格，同时让模型知道原值曾经缺失。

这些是后续特征工程决策。本课只负责证明缺失模式，不进行处理。

## 7. 你的动手任务

练习文件：

```text
scripts/lesson_03c_time_gaps_and_missingness.py
```

请完成三个 TODO：

1. 对按样本分组的倒计时调用 `.diff()`。
2. 用 `==` 逐行比较“平均成交价缺失”和“成交次数为 0”。
3. 用 `.all()` 判断所有行是否一致。

排序、循环打印和小样本读取属于准备工作，已经完整提供。

## 验收标准

- 脚本能读取 `(838, 13)` 的 market 小样本。
- 每个 `sample_id` 的第一条 `time_gap_seconds` 都是计算产生的 `NaN`。
- 能打印 5 个样本的典型间隔、最大间隔和长空档数量。
- 能分别打印平均成交价缺失行数、零成交行数，以及两种模式是否逐行完全一致。
- 不调用 `fillna()`、`dropna()`，也不读取完整 market 文件。
- 能解释为什么 `.diff()` 产生的第一个 `NaN` 与原始字段缺失不是同一种问题。

## 运行后回答

1. 哪两个样本出现了最多的长时间空档？这能否解释它们为什么行数更少？
2. 当前小样本中，`transaction_avgprice` 缺失和 `transaction_count == 0` 是否逐行一致？
3. 为什么现在不能简单删除 `transaction_avgprice` 缺失的行？

## 本课人话总结

数据清洗不是看到洞就立刻补洞。先看这个洞从哪里来：时间序列少了一段，和这一段存在但没有成交，是两件不同的事。只有先把原因分清楚，后面的填补、特征工程和模型输入才不会把真实市场状态改坏。
