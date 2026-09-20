# EXP-TABM-023：seed137 配对的 Relative319 90/10 混合损失

状态：开发折完成；不进入确认折。

只将 seed137、319 特征 TabM 的逐成员 MSE 改为 90% MSE + 10% batch cosine，保留训练 0–49 月、验证 50–59 月、第 15 轮等设置。对照为 EXP-TABM-019 的同 seed、同特征、同折 MSE 模型。

| 指标 | 混合损失相对 MSE |
|---|---:|
| TabM mean | -0.0005804804 |
| TabM trim1 | -0.0008564735 |
| 固定 B001 trim1 | -0.0006677422 |

seed42 开发折曾提高单模 +0.001003，但 seed137 反向，说明这项 loss 改动不稳定。按预先的分阶段规则停止，不跑 seed137 的 60–70 月确认折。

产物：`scripts/exp_tabm_023_relative319_hybrid10_seed137.py` 和 `data/interim/tree_experiments/EXP-TABM-023-RELATIVE319-HYBRID10-SEED137/comparison.json`。
