# 第 4D 课：特征重要性与单变量实验

## 1. 本课目标

上一课已经得到一个可信的 V1 基线：月份 0～59 训练，月份 60～70 验证，余弦分数约为 `0.077821`。现在遇到的新麻烦是：模型已经训练完成，但 40 个输入特征中，它主要依靠哪些特征做判断？下一步又应该改什么？

本课完成后，你应该能够：

1. 解释 LightGBM 的 `gain importance` 和 `split importance`。
2. 建立、排序并检查一个 40 行的特征重要性表。
3. 知道特征重要性不等于因果关系，也不直接代表删掉特征后的分数变化。
4. 根据结果提出一个只改变一件事的实验假设。

本课不训练新模型，也不扫描参数。第一个改进实验将在下一课单独实现，确保基线和改进之间只有一个主要变量发生变化。

## 2. 配套阅读

本课不安排新的外部阅读。

原因是：ISLP 对树模型变量重要性的介绍主要服务于一般树模型，而本课需要准确区分 LightGBM 接口中的 `gain` 与 `split`。这是库接口与实验解释问题，直接由本讲义结合当前模型讲清楚更合适。Python 和 LightGBM API 文档也按项目约定由 Codex 查证，不把源文档交给你阅读。

读完本讲义后，请带着两个问题运行练习：

1. 一个特征被使用次数最多，是否代表它带来的总改进最大？
2. 某特征重要性很高，是否足以证明它与未来收益存在因果关系？

## 3. 人话直觉：模型到底“用了谁”

LightGBM 中每棵树都会不断提问，例如：

```text
ask_price_1_last <= 某个阈值吗？
book_volume_imbalance_1 <= 某个阈值吗？
```

一次提问叫作一次分裂（split）。好的分裂会让分裂后的训练误差比原来更小。特征重要性就是把许多棵树中的这些分裂统计起来。

### split importance

`split` 统计某个特征一共被用于多少次分裂。

小例子：

- 特征 A 被使用 10 次；
- 特征 B 被使用 3 次。

那么 A 的 `split importance` 更大。但是使用次数多不代表每一次都很有效。

### gain importance

`gain` 累加某个特征带来的训练目标改进量。

小例子：

- A 使用 10 次，每次 gain 约为 1，总 gain 为 10；
- B 使用 3 次，每次 gain 约为 8，总 gain 为 24。

虽然 B 使用次数更少，但它带来的总改进更大。本课主要按照 `gain` 排名，`split` 作为辅助观察。

## 4. 第一次出现的 LightGBM 接口

### 4.1 `Booster`

它解决的问题：把上一课保存的 LightGBM 模型重新载入内存，不必再次训练。

- 来源：LightGBM，需要 `from lightgbm import Booster`。
- 基本语法：`model = Booster(model_str=model_text)`。
- 输入：`model_str` 是包含完整模型内容的 Python 字符串。
- 返回：一个 `Booster` 模型对象。
- 当前用途：读取 `lightgbm_market_v1.txt`，然后查询其特征名和重要性。
- 常见错误：直接把当前中文路径传给某些 LightGBM C++ 文件接口可能失败，所以练习先用 Python 的 `read_text()` 读取，再传入 `model_str`。

最小例子只展示结构：

```python
from lightgbm import Booster

model_text = model_path.read_text(encoding="utf-8")
model = Booster(model_str=model_text)
```

### 4.2 `model.feature_name()`

它解决的问题：取得训练模型时使用的全部特征名。

- 它是 `Booster` 对象的方法，不需要额外导入。
- 基本语法：`feature_names = model.feature_name()`。
- 输入：没有额外参数。
- 返回：Python `list`，其中每个元素是一个特征名字符串。
- 当前模型返回长度应为 40。
- 常见错误：方法需要圆括号；`model.feature_name` 只是方法对象，`model.feature_name()` 才真正调用它。

### 4.3 `model.feature_importance()`

它解决的问题：取得每个特征的 LightGBM 重要性数值。

基本语法：

```python
gain_values = model.feature_importance(importance_type="gain")
split_values = model.feature_importance(importance_type="split")
```

- 输入 `importance_type`：字符串，常用值是 `"gain"` 或 `"split"`。
- 返回：形状为 `(特征数,)` 的一维 NumPy 数组。
- 数组顺序与 `model.feature_name()` 完全对应。
- `gain` 通常是浮点数；`split` 通常是整数次数。
- 常见错误：分别排序名称和数值会破坏对应关系。应该先放进同一个 DataFrame，再按整行排序。

## 5. 建表、归一化和排序

我们先把三个等长对象组成一张表：

```python
example_table = pd.DataFrame(
    {
        "feature": ["A", "B"],
        "gain": [10.0, 30.0],
        "split_count": [8, 3],
    }
)
```

返回的是形状 `(2, 3)` 的 DataFrame。字典 `{}` 的键变成列名，每个列表 `[]` 变成一列；各列长度必须相同。

原始 gain 不容易直观看出占比，因此转换为百分比：

```python
total_gain = example_table["gain"].sum()
example_table["gain_percentage"] = (
    example_table["gain"] / total_gain * 100
)
```

这里：

- `example_table["gain"]` 取出一列 Series；
- `.sum()` 把这列全部数值相加，返回一个数；
- Series 除以一个数时，会对每个元素分别相除；
- 新列的总和应接近 100。

最后排序：

```python
sorted_table = example_table.sort_values(
    by="gain_percentage",
    ascending=False,
)
```

`ascending=False` 表示降序，也就是最大的放在最前面。`sort_values()` 返回排序后的新 DataFrame；若不接住返回值，原变量默认不会改变。

## 6. 重要性不能证明什么

看到排名后要保持克制：

1. **不是因果关系。** 模型依赖某列，不等于改变这列就会引起未来收益变化。
2. **相关特征会分摊或争夺重要性。** 买一价和卖一价高度相关，一列先完成了分裂，另一列的重要性可能被压低。
3. **这是训练过程统计。** gain 描述训练目标中的分裂收益，不等于对验证余弦分数的独立贡献。
4. **低重要性不等于可以安全删除。** 真正判断删除是否有益，需要重新训练并在相同验证集上比较。

所以我们把重要性当作“提出实验假设的线索”，而不是最终结论。

## 7. 为什么下一步考虑近期窗口

V1 把约 10 分钟的 market 序列压成全段均值、极值、总和与首尾值。这样很省内存，但可能把离预测时点最近的状态冲淡。

例如，10 分钟平均买卖价差可能较小，但最后 60 秒突然扩大。全段均值无法清楚表达这个变化。下一课候选实验是：

```text
V1 的 40 个全段特征
+ 最后 60 秒的少量聚合特征
```

其他数据、训练月份、验证月份和 LightGBM 参数全部保持不变。这样分数变化才主要归因于“加入近期窗口信息”。是否最终采用这个实验，要先结合你运行得到的重要性排名作判断。

## 8. 你的动手任务

文件：`scripts/lesson_04d_feature_importance.py`

固定代码已经负责路径、模型载入和结果检查。你只完成三个有学习价值的 TODO：

1. 用特征名、gain 和 split 次数建立 DataFrame。
2. 计算 `gain_percentage`，再按它降序排序。
3. 运行后在文件底部用中文回答前两个结果问题；第三题采用下方的多方案判断题。

验收标准：

- 重要性表 shape 为 `(40, 4)`；
- `gain_percentage` 总和接近 100；
- 排名从大到小；
- 你能说出 gain 前三名，并解释为什么重要性只是下一步实验的线索。

第三题改为：根据当前重要性结果，下面哪些下一步行动可能合理，哪一个大概率不合理？可以多选合理项，并分别写一句依据。

- A. 增加最后 60 秒的买一价、卖一价、价差和盘口不平衡统计，再保持其他条件不变重新训练。
- B. 增加第二档盘口价格与挂单量特征，检查更深的订单簿是否提供额外信息。
- C. 直接删除所有 gain 排名靠后的特征，并在不重新验证的情况下认定模型一定会提高。

选择后请分别说明：重要性表提供了什么证据、这个证据有什么局限，以及该方案应如何验证。Codex 在你回答前不公布判断。

完成后告诉我“已完成 4D”。我会先读取并运行你的文件，不直接修改。
