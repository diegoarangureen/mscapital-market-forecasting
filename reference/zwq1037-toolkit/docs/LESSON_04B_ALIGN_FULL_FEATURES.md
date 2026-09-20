# 第 4 课 B：完整特征、标签对齐与训练/验证表

## 本课所处阶段

阶段 4：建立 LightGBM 强基线。

第 4A 课只在 5 个样本上证明了“一段 market 可以聚合成一行”。现在完整训练 market 已经安全聚合完成：

```text
原始 market：221,756,611 行
→ 按列、短进程顺序聚合
→ 完整特征：1,257,637 行 × 41 列
```

41 列包括：

```text
1 个 sample_id
+
40 个模型特征
```

本课把它与 label 按键一对一对齐，再恢复第 2B 课已经学过的月份前向划分。完成后，下一课可以直接训练 LightGBM。

## 本课目标

完成后，你应该能够：

1. 理解完整特征表是怎样在有限内存中生成的。
2. 解释价差、中间价变化、买卖盘不平衡和标准差特征的含义。
3. 使用 `DataFrame.merge()` 按 `sample_id` 对齐，而不是依赖两张表碰巧具有相同行顺序。
4. 使用 `validate="one_to_one"` 让 pandas 主动检查键的唯一性关系。
5. 得到月份 0～59 的训练表和月份 60～70 的验证表。
6. 确认 `sample_id`、`month` 和 `target` 没有进入模型特征 `X`。

本课不做这些事情：

- 不重新讲 LightGBM 原理；
- 不要求你手写完整大文件聚合器；
- 不训练模型；
- 不使用随机划分；
- 不填补剩余 6 个标准差缺失值；
- 不用验证分数反过来挑选大量特征。

## 配套阅读

### 本课不安排新的外部阅读

原因如下：

- `.merge()` 是 pandas 接口，由本讲义完整讲解。
- 月份前向划分已在第 2B 课学过，本课属于真实特征表上的复用，不重复安排 ISLP 阅读。
- 价差、盘口和预测时点概念已在前面课程讨论；本课只讲由这些字段得到的具体特征。
- 树模型已经在上一项目学过。

本课重点是把数据结构和验证边界真正连接起来。

## 1. 完整特征怎样在 16 GB 内存下生成？

完整 market 有 2.217 亿行。虽然 Feather 文件只有约 4.09 GiB，但解压成 13 列宽表会明显超过安全内存。

实际测试得到：

```text
完整 sample_id 单列：进程约占 1.78 GiB
sample_id + 一个特征列并聚合：峰值约 2.94 GiB
```

因此生成脚本采用：

```text
子进程 1：sample_id + seconds → 聚合 → 保存小表 → 释放内存
子进程 2：sample_id + transaction_count → 聚合 → 保存小表 → 释放内存
子进程 3：sample_id + transaction_volume → 聚合 → 保存小表 → 释放内存
……
最后只合并已经缩小到每个 sample_id 一行的小表
```

完整机械代码已经提供：

```text
scripts/build_train_market_features.py
```

你不需要重写它。本课只需要理解两条工程原则：

1. 列式文件可以只投影需要的列。
2. 大数据每一步都应尽早从“逐时间行”缩小为“逐样本行”。

生成结果：

```text
data/processed/train_market_features.feather
shape = (1,257,637, 41)
文件大小约 226 MiB
sample_id 从 0 到 1,257,636，全部唯一
```

## 2. 新增金融特征的含义

完整特征说明保存在：

```text
docs/MARKET_FEATURES_V1.md
```

这里先讲四个最重要的派生特征。

### 第一档平均价差 `spread_1_mean`

第一档卖价通常高于第一档买价：

$$
\text{spread}=\text{ask price}-\text{bid price}
$$

数字例子：

```text
ask = 1.002
bid = 0.998
spread = 1.002 - 0.998 = 0.004
```

价差描述买卖双方当前报价之间的距离。更大的价差可能代表流动性较差或市场不确定性较高，但不能单独当作必然涨跌信号。

### 第一档平均中间价 `mid_price_1_mean`

$$
\text{mid price}=\frac{\text{ask price}+\text{bid price}}{2}
$$

上例中：

$$
\frac{1.002+0.998}{2}=1.000
$$

它用买卖报价的中点近似当前市场价格水平。

### 窗口中间价变化 `mid_price_1_change`

原始行从约 600 秒前排到最接近预测时点，所以：

$$
\text{change}=\text{last mid price}-\text{first mid price}
$$

- 正值：观察窗口内中间价净上升。
- 负值：观察窗口内中间价净下降。

它比整段均值多保留了一部分方向信息，但仍不知道中间每一步怎样波动。

### 第一档总量不平衡 `book_volume_imbalance_1`

$$
\text{imbalance}=\frac{\text{bid volume}-\text{ask volume}}
{\text{bid volume}+\text{ask volume}}
$$

如果买卖量非负，结果通常位于 $[-1,1]$：

- 接近 1：买盘挂单量明显更多。
- 接近 -1：卖盘挂单量明显更多。
- 接近 0：两边总量接近。

这只是订单簿状态特征，不代表更多买盘一定导致未来上涨。

## 3. 标准差为什么还有 6 个缺失？

价格标准差用来描述窗口内波动。样本标准差为：

$$
s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}
$$

例如 `[1,2,3]` 的均值为 2：

$$
s=\sqrt{\frac{(1-2)^2+(2-2)^2+(3-2)^2}{3-1}}=1
$$

如果某个字段只有一个有效观测，$n-1=0$，无法计算样本标准差，所以得到缺失。

完整特征表中：

- `transaction_avgprice_std` 有 4 个缺失；
- `ask_price_1_std` 有 1 个缺失；
- `bid_price_1_std` 有 1 个缺失；
- 合计 6 个缺失，没有浮点 `NaN` 或无穷值。

第一版 LightGBM 可以保留这些缺失。不要为了让表格“看起来整齐”就武断填 0。

## 4. 为什么不能按行号直接拼接？

假设特征表是：

```text
sample_id  feature
10         0.8
20         0.3
```

标签表碰巧顺序相反：

```text
sample_id  target
20         1.2
10        -0.4
```

如果只按行位置拼接，会错误地得到：

```text
sample 10 → target 1.2   # 错误
sample 20 → target -0.4  # 错误
```

正确方法是按 `sample_id` 查找对应关系。这就是键连接（key-based join/merge）。

## 5. `DataFrame.merge()`

### 它解决什么问题

根据共同键把两张 DataFrame 的列对齐到同一行，不依赖原始行顺序。

### 来源与导入

它是 pandas `DataFrame` 的方法。项目已经导入：

```python
import pandas as pd
```

### 基本语法

```python
merged_data = left_data.merge(
    right_data,
    on="key_column",
    how="inner",
    validate="one_to_one",
)
```

主要输入：

- `left_data`：左侧 DataFrame。
- `right_data`：圆括号中的第一个参数，右侧 DataFrame。
- `on`：两边共同的键列名，类型是字符串。
- `how="inner"`：只保留左右两边都存在的键。
- `validate="one_to_one"`：要求左键和右键各自都唯一。

返回：新的 `pd.DataFrame`。两张原表不会被修改。

### 最小例子

```python
features = pd.DataFrame(
    {
        "student_id": [10, 20],
        "study_hours": [3, 5],
    }
)

labels = pd.DataFrame(
    {
        "student_id": [20, 10],
        "score": [90, 70],
    }
)

student_data = features.merge(
    labels,
    on="student_id",
    how="inner",
    validate="one_to_one",
)
```

结果仍按键正确对应：

```text
student_id  study_hours  score
10          3            70
20          5            90
```

### `validate="one_to_one"` 为什么重要？

如果 label 中意外出现两个相同 `sample_id`，普通 merge 可能把一行特征复制成多行，训练样本数量悄悄膨胀。

加入验证后，pandas 会抛出 `MergeError`，主动阻止错误结果。它不是修复重复键，而是把错误尽早暴露出来。

### 当前比赛中的作用

```python
model_data = feature_data.merge(
    label_data,
    on="sample_id",
    how="inner",
    validate="one_to_one",
)
```

预期：

```text
feature_data.shape = (1,257,637, 41)
label_data.shape   = (1,257,637, 3)
model_data.shape   = (1,257,637, 43)
```

合并后列数不是 44，因为共同的 `sample_id` 只保留一份。

### 常见错误

- 写成 `on="target"`：target 不是对齐键。
- 省略合并后的行数检查：inner join 可能因为键缺失而减少行数。
- 使用 `concat(axis=1)`：它主要按索引位置对齐，不能替代明确的业务键检查。

## 6. 哪些列不能进入 X？

完整特征表有 41 列，其中第一列是 `sample_id`。固定代码使用：

```python
feature_columns = feature_data.columns[1:].tolist()
```

分步理解：

1. `feature_data.columns`：返回所有列名组成的 pandas `Index`。
2. `[1:]`：从编号 1 开始取到最后，排除编号 0 的 `sample_id`。
3. `.tolist()`：把列名转换成普通 Python 列表。

最终得到 40 个模型特征。以下三列都不进入 `X`：

- `sample_id`：只是对齐标识。
- `month`：本课只用于时间划分，第一版不作为模型特征。
- `target`：预测答案，绝不能作为输入。

## 7. 复用月份前向划分

第 2B 课已经学习过：

```text
月份 0～59  → 训练
月份 60～70 → 验证
```

本课在真实 40 特征表上复用同一规则：

```python
train_condition = model_data["month"] <= 59
valid_condition = model_data["month"] >= 60
```

预期形状：

```text
X_train = (1,064,163, 40)
X_valid = (193,474, 40)
```

这里不再使用随机划分，因为测试月份全部晚于训练月份。

## 8. 你的动手任务

练习文件：

```text
scripts/lesson_04b_align_full_features.py
```

请完成三个 TODO：

1. 使用 `.merge()` 按 `sample_id` 做 inner、一对一连接。
2. 创建训练月份条件 `month <= 59`。
3. 创建验证月份条件 `month >= 60`。

40 个特征名提取、`X/y` 拆分、shape 和泄露检查已经提供并加注释，不要求机械重写。

## 验收标准

- 特征、label 和合并表分别为 `(1,257,637, 41)`、`(1,257,637, 3)`、`(1,257,637, 43)`。
- inner merge 后没有丢失或复制 `sample_id`。
- `X` 正好包含 40 个特征，不含 `sample_id`、`month` 和 `target`。
- `X_train.shape == (1,064,163, 40)`。
- `X_valid.shape == (193,474, 40)`。
- 训练月份最大值小于验证月份最小值。
- 不调用随机划分，不进行模型训练。

## 运行后回答

1. 为什么必须按 `sample_id` merge，而不能相信两张表当前行顺序相同？
2. `validate="one_to_one"` 会帮我们发现什么错误？
3. 为什么 `month` 在本课用于划分，却不放入第一版 `X`？

## 本课人话总结

特征工程完成后，最危险的错误不是模型参数差一点，而是把别人的答案接到自己的特征上。按 `sample_id` 一对一连接，就是给每张特征摘要卡找到唯一正确的答案。再按月份把过去和未来分开，我们才真正拥有一场可以相信的模型考试。
