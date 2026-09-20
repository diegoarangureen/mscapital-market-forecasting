# EXP-GRU-007：静态条件门控 Joint-GRU 开发折

日期：2026-09-13  
状态：开发折完成；暂不进入确认折。

## 目的

提高 Joint-GRU 单模能力。保持 319 静态特征、StrongGRU 编码器、四路 pooling、训练切分、seed42、MSE、学习率和 6 轮训练不变，只增加一个内部交互：静态表示生成 96 维 gate，把序列表示逐通道缩放到原值的 0.5～1.5 倍。

训练 0–49 月，验证 50–59 月。GPU 训练，CPU 最多 2 线程。

## 结果

| epoch | Gated Joint-GRU 单模 | 90% B001 + 10% 模型 |
|---:|---:|---:|
| 4 | 0.144238 | 0.154706 |
| 6 | 0.148232 | 0.155285 |

原 seed42 Joint-GRU 第 6 轮单模为 0.147890，因此 gate 提高约 0.000342。同期 kernel-5 Conv-GRU 单模为 0.148691，仍比门控模型高约 0.000460。

第 6 轮门控模型的 B001 90/10 融合低于原 Joint-GRU 第 6 轮约 0.000109，说明这次单模小增益没有带来更强的互补性。

## 决策

简单通道 gate 有正信号，但增益不足以承担第二个 seed 和确认折成本。保留结果，不扩大实验。若以后继续内部融合，应测试带残差校正的 FiLM 或同时调节 pooling 分支，而不是搜索当前 gate 的宽度和权重。

## 产物

- `scripts/exp_gru_007_gated_joint_dev.py`
- `data/interim/sequence_experiments/EXP-GRU-007-GATED-JOINT-DEV/result.json`
- `data/interim/sequence_experiments/EXP-GRU-007-GATED-JOINT-DEV/joint_gated_gru319/result.json`

