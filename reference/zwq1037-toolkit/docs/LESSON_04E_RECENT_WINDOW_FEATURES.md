# 第 4E 课：最后 60 秒窗口特征

## 1. 本课要解决什么问题

Market V1 把预测前约 600 秒的数据压缩成一行。这样建立了可靠基线，但全段平均可能掩盖临近预测时点的市场变化。

第 4D 课中，gain 排名前两位的是：

1. `ask_price_1_last`
2. `bid_price_1_last`

这给了我们一个线索：越接近预测时点的盘口状态可能值得单独描述。因此本课只改变一个主要变量：在原有 40 个 V1 特征之外，增加最后 60 秒的聚合特征。

本课完成后，你应该能够：

- 根据倒计时字段正确选出最后 60 秒，而不是最早 60 秒；
- 解释“全段特征”和“近期窗口特征”保留的信息差异；
- 设计公平对照：只增加窗口特征，其他条件保持不变；
- 根据验证分数判断假设是否得到支持。

## 2. 配套阅读

本课不安排新的外部阅读。

原因是 `seconds_before_predict` 是本比赛特有的相对预测时间，ISLP 和 Quant Wiki 没有能直接解释这套字段与窗口筛选关系的对应小节。这里最重要的是在真实数据上判断窗口方向，并完成一次受控实验，由本讲义完整承担。

## 3. 什么叫“最后 60 秒”

`seconds_before_predict` 表示距离预测时点还剩多少秒：

```text
600 → 距离预测还很远
120 → 距离预测还有两分钟
 60 → 距离预测还有一分钟
  5 → 马上到预测时点
```

所以最后 60 秒的条件是：

```python
seconds_before_predict <= 60
```

不是 `>= 60`。

数字例子：某个样本有四行，其倒计时为：

```text
[120, 58, 30, 2]
```

应用 `<= 60` 后留下：

```text
[58, 30, 2]
```

这些才是离预测时点最近的记录。

## 4. 窗口特征保留了什么

假设一段 600 秒序列的价差如下：

```text
前 540 秒：大多为 0.001
最后 60 秒：突然扩大到 0.004
```

全段平均会被大量早期记录拉低，可能只有约 `0.0013`。最后 60 秒平均则接近 `0.004`，能更清楚描述预测前的近期状态。

窗口特征并不一定提高分数，因为：

- 最近 60 秒可能真的含有更强信号；
- 也可能只是噪声；
- V1 的 `last`、`min`、`max` 已经包含部分重叠信息；
- 固定 60 秒未必是最合适的窗口长度。

因此它是待验证假设，不是保证有效的技巧。

## 5. 本课使用的现有语法

这部分不引入新的 pandas API，只复用你已经使用过的工具：

- `<=`：逐行比较，得到布尔 Series；
- `.loc[条件, 列名]`：根据布尔条件选择行和列；
- `.copy()`：让小窗口表独立，便于新增分析列；
- `groupby().agg()`：把每个样本的多行窗口压成一行；
- `reset_index()`：把分组键恢复为普通列。

### 最小无关例子

```python
import pandas as pd

weather_data = pd.DataFrame(
    {
        "city": ["A", "A", "B", "B"],
        "minutes_before_noon": [90, 20, 80, 10],
        "temperature": [18, 22, 25, 28],
    }
)

# 选择中午前最后30分钟的记录。
# Select records from the final 30 minutes before noon.
recent_condition = weather_data["minutes_before_noon"] <= 30
recent_weather = weather_data.loc[recent_condition].copy()

recent_summary = (
    recent_weather.groupby("city")
    .agg(
        recent_temperature_mean=("temperature", "mean"),
        recent_row_count=("temperature", "size"),
    )
    .reset_index()
)
```

输入 `weather_data` 的 shape 是 `(4, 3)`；过滤后 `recent_weather` 是 `(2, 3)`；聚合结果是每个城市一行，shape 为 `(2, 3)`。

## 6. 公平对照必须保持什么不变

最终比较 EXP-001 与窗口版本时，保持以下内容不变：

- 训练月份：0～59；
- 验证月份：60～70；
- 原有 40 个 V1 特征全部保留；
- LightGBM 参数相同；
- early stopping 仍使用验证余弦；
- 评分函数相同。

唯一主要变化是加入最后 60 秒特征。这样若分数变化，我们才有理由把它主要归因于新窗口信息。

## 7. 先判断，再运行

下面三种说法中，有两个是可能发生的结果，一个推理大概率不成立。先选出你认为不成立的一个，再说明你更倾向于另外哪种结果：

- A. 分数提高，因为近期窗口补充了全段聚合没有清楚表达的状态。
- B. 分数不变或下降，因为近期特征与 V1 高度重复，或者引入了更多噪声。
- C. 分数必然提高，因为 `ask_price_1_last` 和 `bid_price_1_last` 的 gain 排名前两位。

不要在实际训练前寻找标准答案。A 和 B 都是在数据上可能发生的结果，关键是识别“可能”与“必然”的区别。

## 8. 你的动手任务：4E-A

文件：`scripts/lesson_04e_recent_window_small.py`

这一步只使用已经安全保存的 5 个样本，共 838 行，运行几乎瞬间完成。你亲手完成：

1. 写出最后 60 秒的布尔条件；
2. 用 `.loc` 筛出窗口数据；
3. 计算逐行第一档价差；
4. 按 `sample_id` 聚合为每个样本一行。

验收标准：

- 所有保留行都满足 `seconds_before_predict <= 60`；
- 5 个样本都存在窗口记录；
- 聚合表 shape 为 `(5, 5)`；
- 能解释为什么条件是 `<= 60`。

4E-A 通过后，Codex 提供完整大数据窗口构建器；你不需要机械重写分块读取和子进程代码。随后在同一课的 4E-B 中训练窗口版本，与 `0.077821` 公平对照。

## 9. 4E-B：完整构建与公平重训

4E-A 已验收后使用以下两个固定脚本：

- `scripts/build_train_market_last60_features.py`：逐列、短子进程读取完整 market，先筛选 `seconds_before_predict <= 60`，再生成22个窗口特征并保存为 `data/processed/train_market_last60_features_complete.feather`。
- `scripts/lesson_04e_train_last60_lightgbm.py`：把40个V1特征与22个窗口特征一对一连接，并复用EXP-001的划分、参数、早停与评分。

大文件构建代码由Codex提供，因为这里主要是已经学过的重复聚合和内存工程，不值得机械抄写。你需要亲手做的是：

1. 在训练脚本的“运行前判断”处写下预测；
2. 运行完整特征构建器；
3. 运行公平对照训练；
4. 根据分数变化判断结果更支持A还是B，并说明为什么一次实验不能证明60秒是最佳窗口。

运行顺序：

```powershell
D:\anaconda\envs\pytorch\python.exe scripts\build_train_market_last60_features.py
D:\anaconda\envs\pytorch\python.exe scripts\lesson_04e_train_last60_lightgbm.py
```

请在项目目录 `F:\深度学习\projects\mscapital_market_forecasting` 下运行。构建器会复用已经完成的中间列，意外中断后再次运行即可继续。

预计最终窗口特征表为每个 `sample_id` 一行；增加的是22列，不增加训练样本行数。验收时比较：

```text
EXP-001：40个V1特征，cosine 0.077821
EXP-002：40个V1特征 + 22个last60特征，等待运行
```
