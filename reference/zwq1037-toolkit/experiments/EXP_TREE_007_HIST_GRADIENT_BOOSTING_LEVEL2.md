# EXP-TREE-007：HistGradientBoosting 加入第二档盘口

## 实验假设与唯一主要变量

第一档盘口只描述最优买卖报价，第二档价格和挂单量可能补充更深一层的流动性与买卖压力。保持 EXP-TREE-005 的模型、target处理、时间划分和62个已有特征不变，只增加35个与第一档同口径的第二档全段及最后60秒特征。

## 设置

- 基线：EXP-TREE-005，62特征 HistGradientBoosting，cosine `0.1061823145`
- 新模型：97特征；新增35个第二档盘口特征
- 训练月份：0～59，1,064,163行
- 验证月份：60～70，193,474行
- target：只减训练期均值，不使用验证标签校准
- 模型：400轮、学习率0.03、31个叶节点、最小叶100、L2=1、关闭内部随机early stopping、种子42
- 训练耗时：36.22秒

## 总体结果

- forward-validation cosine：**0.1093915648**
- 相对 EXP-TREE-005：**+0.0032092503**
- 相对原 EXP-002（0.1038700392）：**+0.0055215256**

第二档特征对整体方向有效，成为当前最佳单模型；尚未进行模型融合。

## 逐月水平与波动

| 指标 | EXP-TREE-005 | EXP-TREE-007 | 变化 |
|---|---:|---:|---:|
| 月度cosine宏观均值 | 0.1022276008 | 0.1052082247 | +0.0029806239 |
| 月度cosine总体标准差 | 0.0149939402 | 0.0160567544 | +0.0010628143 |
| 月度最大值减最小值 | 0.0592855031 | 0.0499078587 | -0.0093776445 |
| 最差月cosine | 0.0737421512（68月） | 0.0822249294（61月） | +0.0084827782 |
| 负分月份数 | 0 | 0 | 0 |

逐月配对中5个月改善、6个月下降，中位变化为 `-0.0008948663`。总体提升主要来自64、67、68月，尤其68月提升 `+0.0185809063`。因此：

- 月均值、最差月和极差改善；
- 标准差和平均绝对离差略增；
- 不能概括为“每个月都更稳”，应记录为整体更强但月份收益分布更不均匀。

逐月明细和配对差值位于 `data/interim/tree_experiments/EXP-TREE-007/monthly_cosine.csv` 与 `monthly_comparison.csv`。

## 产物

- 配置：`data/interim/tree_experiments/EXP-TREE-007/config.json`
- 模型：`outputs/models/hist_gradient_boosting_target_centered_level2.joblib`
- 验证预测：`outputs/predictions/hist_gradient_boosting_target_centered_level2_valid.feather`
