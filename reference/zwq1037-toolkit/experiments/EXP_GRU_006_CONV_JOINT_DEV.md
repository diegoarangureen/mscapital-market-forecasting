# EXP-GRU-006：kernel-5 残差卷积 + Joint-GRU 开发折

日期：2026-09-13  
状态：开发折完成；停止，不进入确认折。

## 目的

检验 GRU 前增加一层局部卷积，能否先识别相邻事件的短期形状，再由 GRU 汇总较长变化。相对原 Joint-GRU 只改变一项：在 input projection 后加入 `Conv1d(kernel_size=5)`，经过 SiLU 后与原表示残差相加，再送入 GRU。padding mask、319 静态特征、四路 pooling、seed42 和训练切分均不变。

训练 0–49 月、验证 50–59 月；共 6 轮，在第 4、6 轮保存预测。GPU 训练，CPU 最多 2 线程。

## 单模结果

| epoch | Conv-GRU cosine | 90% B001 + 10% Conv-GRU delta |
|---:|---:|---:|
| 4 | 0.143797 | +0.000332 |
| 6 | 0.148691 | +0.001034 |

第 6 轮比原 seed42 Joint-GRU 第 6 轮的 0.147890 高约 0.000800，说明局部卷积对单模有小幅帮助。

## 当前最佳完整配方检验

当前 Public 0.135 配方在本地精确重建为 0.155440。候选固定为 90% 原 tree basket、5% 双 seed Joint-GRU、5% seed42 Conv-GRU，没有搜索权重。

| epoch | 完整融合 cosine | 相对当前配方 | 与双 seed Joint-GRU 相关性 |
|---:|---:|---:|---:|
| 4 | 0.155223 | -0.000216 | 0.9184 |
| 6 | 0.155564 | +0.000125 | 0.9350 |

第 6 轮的最差月份相对当前配方下降 0.000103，月度第一四分位数提高 0.000183。总体增益低于预先设定的 `+0.0003` 开发门槛。

## 决策

Conv 前端改善了单模，但它与现有 GRU 高度相关，放入最佳融合后的新增价值太小。停止，不运行 60–70 月确认折，也不增加 seed 或搜索卷积宽度、kernel 和融合权重。

## 产物

- `scripts/exp_gru_006_conv_joint_dev.py`
- `scripts/analyze_conv_gru_dev_blend.py`
- `data/interim/sequence_experiments/EXP-GRU-006-CONV-JOINT-DEV/joint_conv_gru319/result.json`
- `data/interim/sequence_experiments/EXP-GRU-006-CONV-JOINT-DEV/blend_result.json`
- `data/interim/sequence_experiments/EXP-GRU-006-CONV-JOINT-DEV/blend_summary.csv`

