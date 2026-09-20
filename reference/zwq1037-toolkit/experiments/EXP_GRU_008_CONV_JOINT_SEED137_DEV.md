# EXP-GRU-008：Conv-GRU kernel5 的 seed137 配对开发折

日期：2026-09-13。状态：完成；未通过两 seed 晋级门槛。

相对 EXP-GRU-006，只将 seed42 改为 seed137。保持 319 静态特征、kernel5 残差卷积、GRU96、四路 pooling、MSE、训练 0–49 月、验证 50–59 月和第 6 轮评估不变。与相同 seed 的原 Joint-GRU 比较。

| seed | Joint-GRU 单模 | Conv-GRU 单模 | 单模差值 | 90/10 固定融合差值 |
|---:|---:|---:|---:|---:|
| 42 | 0.147890 | 0.148691 | +0.000801 | +0.000033 |
| 137 | 0.144806 | 0.142974 | -0.001833 | -0.000273 |

两个 seed 的单模平均差值为 -0.000516。预先约定的晋级条件为两 seed 都正向且平均至少 +0.0005，因此停止，不运行 0–59→62–70 确认折。seed42 的单折改善不能作为稳定单模升级依据。

产物：`scripts/exp_gru_008_conv_joint_seed137_dev.py`、`scripts/analyze_conv_gru_seed137_dev.py`、`data/interim/sequence_experiments/EXP-GRU-008-CONV-JOINT-SEED137-DEV/paired_dev_comparison.json`。
