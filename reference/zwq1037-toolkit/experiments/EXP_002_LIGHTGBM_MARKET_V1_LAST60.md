# EXP-002：LightGBM Market V1 + 最后60秒窗口

## 实验假设

Market V1 的全段聚合可能冲淡临近预测时点的状态。保持其他条件不变，加入最后60秒窗口特征，检验近期盘口与成交信息是否提供额外预测信号。

## 实验设置

- 日期：2026-08-30
- 基础特征：40个Market V1全序列聚合特征
- 新增特征：22个最后60秒窗口特征
- 总特征数：62
- 训练月份：0～59
- 验证月份：60～70
- 训练样本数：1,064,163
- 验证样本数：193,474
- 模型与参数：与EXP-001相同的`LGBMRegressor`
- 最佳轮数选择：验证集cosine，early-stopping patience 200

## 数据完整性

- 窗口特征表shape：`(1,257,637, 23)`
- `sample_id`全部唯一
- 5个样本在最后60秒没有market记录：`last60_row_count=0`，其他窗口统计保留缺失
- 没有因窗口为空而删除训练样本

## 正式结果

- 最佳轮数：411
- 训练耗时：26.75秒
- forward-validation cosine：0.1038700392
- EXP-001 cosine：0.0778208601
- 绝对变化：+0.0260491791
- Kaggle分数：尚未提交

## 无效试跑说明

第一次试跑使用窗口表作为inner-join一侧，误删了5个没有最后60秒记录的训练样本，得到约0.104092和953轮。该结果不作为正式实验；修正为以V1完整ID做左连接后得到上述正式结果。

## 结论

结果支持“最后60秒窗口包含V1全段聚合未充分表达的信息”。但一次60秒实验不能证明60秒是最佳窗口，也不能证明单个窗口特征与target存在因果关系。后续传统模型不继续扫描大量窗口；完成LightGBM数学原理课后冻结该基线，进入PyTorch序列模型。

## 产物

- 特征：`data/processed/train_market_last60_features_complete.feather`
- 模型：`outputs/models/lightgbm_market_v1_last60.txt`
- 验证预测：`outputs/predictions/lightgbm_market_v1_last60_valid.feather`
