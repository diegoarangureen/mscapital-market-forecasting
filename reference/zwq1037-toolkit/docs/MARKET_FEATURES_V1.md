# Market 聚合特征 V1

## 产物

- 文件：`data/processed/train_market_features.feather`
- shape：`(1,257,637, 41)`
- 粒度：每个 `sample_id` 一行
- 键：`sample_id`，范围 `0～1,257,636`，全部唯一
- 输入：只使用预测时点前的训练 market，不读取 label 或 target
- 构建脚本：`scripts/build_train_market_features.py`

## 安全生成方式

原始 market 有 `221,756,611` 行。生成器每次只读取 `sample_id` 和一个源特征列，在独立子进程中聚合后写出小表。实测单次峰值约 `2.94 GiB`；每个子进程结束后释放内存，最后只连接逐样本结果。

## 特征分组

| 特征组 | 特征 | 含义 |
|---|---|---|
| 样本完整性 | `market_row_count` | 实际 market 时间步数 |
| 时间覆盖 | `seconds_max`, `seconds_min`, `seconds_coverage` | 最早、最近和覆盖秒数 |
| 时间空档 | `long_gap_count` | 相邻间隔超过 4.5 秒的数量；4.5 是当前观察阈值 |
| 成交次数 | `transaction_count_sum`, `transaction_count_mean`, `transaction_count_max` | 总次数、平均次数和单段峰值 |
| 成交覆盖 | `has_transaction_ratio` | 有成交时间段占比 |
| 成交量 | `transaction_volume_sum`, `transaction_volume_mean`, `transaction_volume_max` | 总量、平均量和峰值 |
| 平均成交价 | `transaction_avgprice_mean`, `_std`, `_min`, `_max` | 有成交时间段的价格水平与波动范围 |
| 成交价端点 | `transaction_avgprice_first_valid`, `_last_valid` | 最早和最近的有效平均成交价 |
| 第一档卖价 | `ask_price_1_mean`, `_std`, `_min`, `_max`, `_first`, `_last` | 卖价水平、波动、范围与端点 |
| 第一档买价 | `bid_price_1_mean`, `_std`, `_min`, `_max`, `_first`, `_last` | 买价水平、波动、范围与端点 |
| 第一档卖量 | `ask_volume_1_sum`, `_mean`, `_max` | 卖盘总深度、平均深度和峰值 |
| 第一档买量 | `bid_volume_1_sum`, `_mean`, `_max` | 买盘总深度、平均深度和峰值 |
| 价差与中间价 | `spread_1_mean`, `mid_price_1_mean`, `mid_price_1_change` | 平均买卖价差、平均中间价和首尾净变化 |
| 盘口不平衡 | `book_volume_imbalance_1` | `(买量总和-卖量总和)/(买量总和+卖量总和)` |

## 缺失情况

特征表共有 6 个 null：

| 特征 | null 数 | 原因 |
|---|---:|---|
| `transaction_avgprice_std` | 4 | 对应样本只有一个有效成交均价，样本标准差无定义 |
| `ask_price_1_std` | 1 | 对应样本只有一个 market 行 |
| `bid_price_1_std` | 1 | 对应样本只有一个 market 行 |

V1 不填补这些值；LightGBM 基线保留缺失。未发现浮点 `NaN`。

## V1 明确没有做什么

- 没有使用第二档盘口字段。
- 没有加入 order 和 transaction 原始事件流。
- 没有生成不同时间窗口特征，例如最后 60 秒或最后 15 秒。
- 没有使用 target 选择特征。
- 没有标准化；树模型第一版不需要。

这些内容应作为后续独立实验逐项加入，避免一次改变太多因素。
