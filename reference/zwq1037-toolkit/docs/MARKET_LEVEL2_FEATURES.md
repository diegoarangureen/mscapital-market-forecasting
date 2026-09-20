# 第二档盘口聚合特征

## 产物与边界

- 文件：`data/processed/train_market_level2_features.feather`
- shape：`(1,257,637, 36)`，即 `sample_id` 加35个新特征
- 原始字段：`ask_price_2`、`bid_price_2`、`ask_volume_2`、`bid_volume_2`
- 输入只来自预测时点前的训练 market；没有读取 label 或 target
- 构建脚本：`scripts/build_train_market_level2_features.py`
- 每个子进程只读取 `sample_id`、`seconds_before_predict` 和一个第二档字段，共扫描四次原始 market；中断后可复用已完成的分列部件

## 特征设计

本次只做第一档特征的镜像，避免同时改变统计口径。

| 范围 | 第二档价格 | 第二档挂单量 | 派生特征 |
|---|---|---|---|
| 全段 | mean、std、min、max、first、last | sum、mean、max | 平均价差、中间价均值、中间价首尾变化、盘口量不平衡 |
| 最后60秒 | mean、std、start、end | mean | 平均价差、中间价首尾变化、盘口量不平衡 |

共新增35列。第二档不平衡定义为：

$$
\frac{\text{bid volume}_2-\text{ask volume}_2}
{\text{bid volume}_2+\text{ask volume}_2}。
$$

分母为0时保留缺失，不制造无意义的0。

## 完整性检查

- `sample_id` 共1,257,637个，全部唯一，并与Market V1逐行一致。
- 所有数值列均无正负无穷。
- 全段价格标准差各有1个缺失，对应只有一行market记录的样本。
- 5个样本在最后60秒没有market记录，因此相应近期第二档统计为缺失。
- 最后60秒价格标准差各有82个缺失：包括5个空窗口以及只有一个窗口观测、无法定义样本标准差的样本。

5个真实样本的smoke检查核对了均值、价差、首尾顺序、最后60秒筛选、ID唯一性和派生不平衡。Polars与pandas对float32均值的不同累加顺序造成最大约 `1.19e-7` 的差异，属于浮点精度而不是聚合错误。
