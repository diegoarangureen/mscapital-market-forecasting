# 第 3 课 B：看懂一个完整 market 样本

## 本课所处阶段

阶段 3：安全读取并对齐 market 小样本。

第 3A 课只看了大文件的目录结构。本课真正打开 5 个已经安全导出的完整样本，观察一条样本内部的时间顺序，并确认它怎样对应标签表中的一行。

## 本课目标

完成后，你应该能够：

1. 解释为什么 market 中一个 `sample_id` 会占多行，而 label 中只占一行。
2. 使用 `groupby(...).size()` 统计每个样本有多少个时间步。
3. 使用 `sort_values()` 按距离预测时点的秒数排序。
4. 从表中选出一个完整样本，并找到它对应的 `target`。

本课不做这些事情：

- 不训练模型；
- 不读取 4.09 GiB 的原始 market 文件；
- 不填补缺失值；
- 不把一整段序列汇总成模型特征；
- 不使用 `merge()`，避免一课同时引入太多新工具。

## 配套阅读

### 已有前置知识：本课不重复阅读限价单簿

第 1A 课已经阅读过 Quant Wiki 的“限价单”和“限价单簿”，也已经讨论过买方、卖方和订单簿。本课直接复用这些知识，不再重复安排同一页面。

本课真正新增的是“如何把混在一张表里的多条序列分开”和“如何恢复一条序列的阅读顺序”。这些属于 Python/pandas 接口知识，由本讲义直接讲解，不要求学习者阅读官方 API 文档。

### 本课无需课前阅读

本课没有适合学习者额外阅读的教材章节：

- `groupby()`、`.size()` 和 `sort_values()` 是 pandas 接口，由 Codex 根据官方文档核对后，在本讲义中用直觉、参数、返回值和最小例子完整讲解。
- “一个 `sample_id` 的多行 market 对应 label 中一个 `target`”是本比赛特有的数据组织方式，由讲义结合实际小样本解释。
- ISLP 和《动手学深度学习》还没有进入与本课直接相关的统计模型或神经网络内容。

因此，本课把原本的阅读时间用于先读讲义第 1～4 节，然后回答：

1. `groupby("sample_id")` 做完后，为什么还没有得到最终行数？
2. 本课为什么用 `.size()` 而不是 `.count()` 统计时间步？
3. `seconds_before_predict` 从大到小排列时，为什么是在走向预测时点？

## 1. 我们遇到的麻烦是什么？

label 表看起来很简单：

```text
sample_id    month    target
0            0        0.005409
1            0        0.000610
```

每个 `sample_id` 只有一个预测目标。

但 market 表记录的是预测发生前的一段过程。以同一个样本为例：

```text
sample_id    seconds_before_predict    ask_price_1    bid_price_1
0            596.0                     ...            ...
0            593.0                     ...            ...
0            590.0                     ...            ...
...          ...                       ...            ...
0              2.0                     ...            ...
```

所以两张表不是“一行对一行”，而是：

```text
market 中多行时间序列  ── sample_id ──>  label 中一个 target
```

以后模型要做的事，就是从这段预测前的市场变化中提取信息，预测那一个 `target`。

## 2. `DataFrame.groupby()`

### 它解决什么问题

假设一张表混在一起记录了许多学生：

```text
name    score
小明    80
小明    90
小红    70
```

如果想分别处理小明和小红，需要先按照 `name` 把行分组。`groupby()` 就是做这件事。

它的直觉通常称为 split-apply-combine：

```text
按键拆成组（split）
→ 每组执行同一种操作（apply）
→ 组合各组结果（combine）
```

### 来源与导入

它是 pandas `DataFrame` 对象的方法。导入 pandas 后即可使用：

```python
import pandas as pd
```

### 基本语法

```python
grouped_data = data.groupby("column_name")
```

- `data`：一个 `pd.DataFrame`。
- `"column_name"`：用来分组的列名，类型是字符串 `str`。
- 返回值：一个 `DataFrameGroupBy` 对象。

注意：`groupby()` 本身通常还没有算出最终数字。它更像是“已经告诉 pandas 按什么分组，等待下一条操作”。

### 最小例子

```python
students = pd.DataFrame(
    {
        "name": ["小明", "小明", "小红"],
        "score": [80, 90, 70],
    }
)

grouped_students = students.groupby("name")
```

`grouped_students` 不是新的普通 DataFrame，而是一个分组对象。接下来可以问每组有几行。

### 当前比赛中的作用

```python
grouped_market = market_data.groupby("sample_id")
```

这里把 838 行 market 小样本按 `sample_id` 分成 5 组。

### 常见错误

不能把列名写成未定义变量：

```python
# 错误：如果没有名为 sample_id 的 Python 变量，会报 NameError。
market_data.groupby(sample_id)

# 正确：列名是字符串。
market_data.groupby("sample_id")
```

## 3. 分组对象的 `.size()` 方法

### 它解决什么问题

`.size()` 计算每个分组包含多少行。

### 基本语法

```python
row_counts = grouped_data.size()
```

- 输入：不再额外传参数。
- 返回：`pd.Series`。
- Series 的索引是分组键，值是对应组的行数。

### 最小例子

```python
grouped_students = students.groupby("name")
student_row_counts = grouped_students.size()
print(student_row_counts)
```

结果类似：

```text
name
小明    2
小红    1
dtype: int64
```

### 分步写法与紧凑写法

分步写法：

```python
grouped_market = market_data.groupby("sample_id")
row_counts = grouped_market.size()
```

等价的紧凑写法：

```python
row_counts = market_data.groupby("sample_id").size()
```

第二行中的两个点表示连续调用：先由 DataFrame 得到分组对象，再调用该对象的 `size()` 方法。

### `.size()` 和 `.count()` 不一样

- `.size()`：数每组有多少行，即使某个单元格是 `NaN`，这一行仍然存在。
- `.count()`：分别统计每一列中有多少个非缺失值。

本课要问的是“每个样本有多少个时间步”，因此使用 `.size()`。

### 常见错误

这里的 `size` 是分组对象的方法，必须写圆括号：

```python
grouped_market.size()
```

你以前可能见过 `data.size` 属性。对象不同，接口也可能不同；判断时应查看 `type(object)`、`dir(object)` 或官方文档。

## 4. `DataFrame.sort_values()`

### 它解决什么问题

表格在磁盘里的行顺序不应该被我们盲目信任。为了从“离预测较远”看到“离预测较近”，需要明确按照 `seconds_before_predict` 排序。

### 来源与导入

它也是 pandas `DataFrame` 的方法，不需要额外导入。

### 基本语法

```python
sorted_data = data.sort_values(
    by="column_name",
    ascending=False,
)
```

主要参数：

- `by`：按哪一列排序，常用类型是字符串。
- `ascending=True`：从小到大。
- `ascending=False`：从大到小。

返回值是一个排序后的新 `DataFrame`，形状与原表相同。本课不用 `inplace=True`，让原变量和新变量的关系更清楚。

### 最小例子

```python
temperatures = pd.DataFrame(
    {
        "minute": [2, 0, 1],
        "temperature": [22, 20, 21],
    }
)

sorted_temperatures = temperatures.sort_values(
    by="minute",
    ascending=True,
)
```

结果会按分钟 `0, 1, 2` 排列。

### 当前比赛中的作用

`seconds_before_predict` 是“距离预测时点还剩多少秒”。所以：

- 数值较大：更早、更远离预测时点；
- 数值较小：更晚、更接近预测时点。

若按 `ascending=False` 排序，阅读顺序就是：

```text
约 600 秒前 → ... → 约 2 秒前 → 预测时点
```

### 常见错误

不要把“秒数越来越小”误解为时间倒流。这里存的是倒计时，不是普通递增时间戳。

## 5. 为什么样本行数可能不完全相同？

我们准备的 5 个样本分别有大约 100～200 行，而不是全部严格相同。此时只能先观察，不能马上武断地下结论。

可能的原因包括：

- 某些等间隔 bar 没有记录；
- 样本可用的历史区间不同；
- 原始数据经过了过滤。

本课只确认实际行数和时间范围。下一课再检查相邻时间差、缺失位置和是否需要补齐。

`transaction_avgprice` 中出现 `NaN`，很可能表示该时间段没有成交，因此无法计算平均成交价；这是基于字段语义的推断，不应在尚未验证前直接填成 0。

## 6. 在 PyCharm 中运行

练习文件：

```text
scripts/lesson_03b_one_market_sample.py
```

数据文件已准备好：

```text
data/interim/market_sample_ids_0_4.feather
data/interim/label_sample_ids_0_4.feather
```

它们只包含 5 个完整样本。不要把练习路径改回 `data/raw/train/market.feather`。

## 7. 你的动手任务

请只完成三个 TODO：

1. 用 `market_data.groupby("sample_id").size()` 统计每个样本的行数。
2. 用布尔条件选出 `sample_id == selected_sample_id` 的行。
3. 用 `sort_values()` 将这个样本按 `seconds_before_predict` 从大到小排序。

读取路径、打印、标签筛选和缺失值检查已经提供，不需要为了形式机械重写。

运行后，请先自己观察并回答：

1. 5 个样本的时间步数量是否完全相同？
2. 对样本 0 来说，打印结果的第一行和最后一行，哪一行更接近预测时点？为什么？
3. market 中样本 0 的多行，最终对应 label 中几个 `target`？

## 验收标准

- market 小样本的 shape 是 `(838, 13)`，label 小样本的 shape 是 `(5, 3)`。
- `row_counts` 是以 `sample_id` 为索引的 Series，并包含 5 个计数。
- 选出的 DataFrame 只包含 `sample_id == 0`。
- `seconds_before_predict` 按从大到小排列。
- 能解释“一段 market 序列对应一个 target”。
- 没有读取完整原始 market 文件，也没有开始填缺失值。

## 本课人话总结

一个训练样本不是 market 表里的一行，而是同一个 `sample_id` 下的一整段倒计时序列。label 表给这整段过程配一个答案。我们现在做的，是先把“题目的一段输入”和“它的一个答案”正确对应起来；连输入单位都没看清之前，不应该急着训练模型。
