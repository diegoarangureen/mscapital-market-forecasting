# EXP-001：LightGBM Market V1 基线

## 实验设置

- 日期：2026-08-25
- 数据：`train_market_features.feather`
- 粒度：每个 `sample_id` 一行
- 特征：40 个 Market V1 全序列聚合特征
- 训练月份：0～59
- 验证月份：60～70
- 模型：`LGBMRegressor`
- 训练目标：L2 regression
- 最佳轮数选择：验证集 cosine，early-stopping patience 200
- 主要参数：learning rate 0.03、num leaves 31、min child samples 100、reg lambda 1.0

## 结果

- 最佳轮数：356
- 训练耗时：约 21 秒
- forward-validation cosine：0.07782086
- Kaggle 分数：尚未提交

## 产物

- 模型：`outputs/models/lightgbm_market_v1.txt`
- 验证预测：`outputs/predictions/lightgbm_market_v1_valid.feather`

## 当前结论

这是后续实验的固定可信基线。下一次实验保持时间划分和模型参数不变，只增加一组临近预测时点的窗口特征，以判断近期市场状态是否提供额外信息。
