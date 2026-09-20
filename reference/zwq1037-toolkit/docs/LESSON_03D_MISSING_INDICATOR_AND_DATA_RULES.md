# 第 3 课 D：把缺失观察变成明确处理规则

## 本课所处阶段

这是阶段 3“安全读取并对齐 market 小样本”的收尾课。

前面三课已经回答：

```text
market 有哪些列
→ 一个 sample_id 为什么有多行
→ 时间步如何排列
→ 哪些地方存在长空档
→ 平均成交价为什么可能缺失
```

本课把这些零散观察整理成一套后续模型都必须遵守的数据规则。

## 本课目标

完成后，你应该能够：

1. 区分数据事实、样本证据、合理推断和处理决定。
2. 理解什么是缺失指示列（missing indicator）。
3. 创建布尔特征 `has_transactions`，但不覆盖原始成交次数。
4. 解释为什么 LightGBM 和神经网络对 `NaN` 的处理方式不完全相同。
5. 知道本项目为何保留原始行、原始 `NaN` 和缺失含义，直到模型输入阶段再转换。

本课不做这些事情：

- 不训练 LightGBM；
- 不把 `NaN` 立即填成某个数字；
- 不标准化价格；
- 不把 5 个样本的结论假装成完整数据的定律；
- 不要求你机械编写数据字典。

## 配套阅读

### 本课不安排新的外部阅读

原因如下：

- ISLP 第 56 页关于缺失数据的短段落刚在第 3C 课读过，短期内不重复。
- `notna()` 等 pandas 接口由本讲义直接讲解，不要求阅读 API 源文档。
- “保留无成交行、为神经网络准备 mask”是当前比赛的工程决策，需要根据实际字段和模型要求分析，而不是照搬某本教材的一般结论。

本讲义承担本课的全部必要知识。学习者已在上一项目掌握决策树、随机森林、Gradient Boosting、LightGBM 和 early stopping；阶段 4 不重复安排树模型入门阅读，只在遇到尚未掌握的新理论时补充教材。

## 1. 四种不同层次的话

做数据分析时，下面四种话不能混在一起。

### 事实（source fact）

由比赛说明或文件本身直接给出，例如：

```text
market 有 transaction_count 和 transaction_avgprice 两列。
```

### 样本证据（sample evidence）

由我们实际检查的 5 个样本得到，例如：

```text
838 行中有 251 行 transaction_avgprice 缺失；
这 251 行的 transaction_count 都等于 0。
```

### 推断（inference）

对证据作出的合理解释，例如：

```text
没有成交时，平均成交价没有定义，所以保存为 NaN。
```

这个解释很合理，但完整数据仍需验证。

### 处理决定（processing decision）

我们为了后续建模制定的规则，例如：

```text
不删除无成交行；保留 transaction_count；
在需要时添加 has_transactions 指示列。
```

好的项目文档应该让别人看得出一句结论属于哪一层。我们已经把当前认识整理到：

```text
docs/MARKET_DATA_DICTIONARY.md
```

## 2. 什么是缺失指示列？

假设有三段时间：

```text
平均成交价    成交次数
1.01          3
NaN           0
1.02          2
```

可以增加一列：

```text
平均成交价    成交次数    是否有成交
1.01          3           True
NaN           0           False
1.02          2           True
```

`has_transactions` 就是缺失指示列的一种。它把“为什么缺失”变成显式状态：

```python
has_transactions = market_data["transaction_count"] > 0
```

这不会删除或填补任何原始数据，只会生成一个与原表行数相同的布尔 `Series`。

### 为什么不直接覆盖 `transaction_count`？

因为两个特征携带的信息不同：

- `has_transactions`：只回答有或没有。
- `transaction_count`：还回答发生了多少次，例如 1 次和 20 次不同。

所以新列是补充，不是替换。

### 当前代码中的 shape

原 market 小样本是：

```text
(838, 13)
```

增加一列后应为：

```text
(838, 14)
```

行数保持 838，说明没有因为构造指示列丢掉时间步。

## 3. `Series.notna()`

### 它解决什么问题

`.isna()` 判断“是否缺失”，`.notna()` 判断“是否不缺失”。

### 来源与导入

它是 pandas `Series` 的方法。项目已经使用：

```python
import pandas as pd
```

### 基本语法

```python
available_values = values.notna()
```

- 输入对象：一个 `pd.Series`。
- 参数：本课不传参数。
- 返回：布尔 `pd.Series`，shape 与输入相同。

### 最小例子

```python
prices = pd.Series([1.01, None, 1.02])
price_available = prices.notna()
print(price_available)
```

结果：

```text
0     True
1    False
2     True
```

### 当前比赛中的作用

```python
average_price_available = market_data["transaction_avgprice"].notna()
```

随后可以验证两种说法是否逐行一致：

```text
transaction_count > 0
transaction_avgprice 不缺失
```

### 常见错误

不要用下面的方式判断缺失：

```python
market_data["transaction_avgprice"] != None
```

pandas 中的缺失值有专门的判断规则，应使用 `.isna()` 或 `.notna()`。

## 4. 把 Series 加成 DataFrame 的新列

你以前已经见过：

```python
analysis_data["true_label"] = y
```

本课使用同样的语法：

```python
market_data["has_transactions"] = has_transactions
```

方括号中的字符串是新列名，等号右边是一个长度为 838 的 Series。pandas 按索引把每个布尔值放到对应行。

常见错误是右边长度或索引与 DataFrame 不匹配。这里的 Series 直接来自同一个 `market_data`，所以能够逐行对齐。

## 5. 为什么不同模型的处理方式不同？

### LightGBM 阶段

上一项目已经学习并运行过 LightGBM 与 early stopping，本节只说明当前金融数据与旧模板不同的输入规则。LightGBM 可以在建树时处理缺失值，因此阶段 4 的第一版基线可以：

- 保留 `NaN`；
- 保留 `transaction_count`；
- 通过实验判断显式 `has_transactions` 是否增加信息。

我们不需要为了让 LightGBM 运行，就提前把平均成交价填成 0。

### 神经网络阶段

神经网络输入通常是数值 Tensor。`NaN` 进入普通算术后很容易让损失也变成 `NaN`，训练无法继续。因此进入 1D CNN 或 Transformer 前，需要：

```text
保留一列 mask，告诉模型原值是否存在
＋
把数值输入中的 NaN 替换成训练流程规定的安全值
```

这里的 mask 可以来自：

```python
has_transactions
```

具体填什么值要与训练集标准化方案一起决定，不能现在利用全部月份统计一个填充值，否则可能产生时间泄漏。

## 6. 本项目从现在开始遵守的规则

对 `transaction_avgprice`：

1. 原始文件保持只读，不覆盖。
2. 不删除无成交行，因为它仍代表一个市场时间点。
3. 不把原始 `NaN` 武断解释成数据损坏。
4. LightGBM 基线先允许保留 `NaN`。
5. 神经网络阶段必须使用缺失 mask，并在训练流程内部决定安全填充值。
6. 在扩大到完整数据时，重新验证“零成交等价于平均成交价缺失”。

对时间空档：

1. 不假设所有样本严格有 200 行。
2. 表格聚合基线可以使用实际存在的行，并把行数、覆盖时长或长空档数作为候选特征。
3. CNN/Transformer 需要固定长度时，再设计 padding 和 time mask；现在不提前补出虚假市场行。

## 7. 你的动手任务

练习文件：

```text
scripts/lesson_03d_missing_indicator.py
```

请完成两个 TODO：

1. 根据 `transaction_count > 0` 创建 `has_transactions`。
2. 用已经学过的逐行比较和 `.all()`，验证它是否与 `transaction_avgprice.notna()` 完全一致。

增加列、shape 检查和打印已经提供。这次不要求你重写重复代码，也不要求填补 `NaN`。

## 验收标准

- 原始 shape 为 `(838, 13)`，增加指示列后为 `(838, 14)`。
- 行数保持不变。
- `has_transactions` 是布尔类型。
- 能检查它是否与平均成交价非缺失逐行一致。
- 代码中没有 `fillna()`、`dropna()` 或完整 market 读取。
- 能分别说出 LightGBM 阶段和神经网络阶段暂定怎样处理这类缺失。

## 运行后回答

1. 增加 `has_transactions` 后，为什么行数不变、列数加 1？
2. 为什么 `has_transactions` 不能完全替代 `transaction_count`？
3. LightGBM 与神经网络面对这里的 `NaN` 时，暂定规则分别是什么？

## 本课人话总结

清洗数据不是把所有表格都变得“没有 NaN”，而是让数据含义在进入模型时不被破坏。我们保留真实市场行和成交次数，再用一个指示列明确告诉后续流程“这里有没有发生成交”。至于怎样把它喂给不同模型，要服从模型的计算方式和时间验证规则。
