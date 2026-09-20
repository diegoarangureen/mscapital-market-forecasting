# EXP-TRANSFORMER-006/007：Joint-Transformer 组合 loss

日期：2026-09-13

保持 Relative319 静态分支、market 事件序列、2 层 4 头 `d_model=96`、seed42 和 batch 设置不变，只把 MSE 换成 `0.35 SmoothL1 + 0.65 centered batch cosine`。开发窗固定选择第 5 轮，再用于确认窗与全量训练。

| 窗口 | 原 MSE Transformer | 组合 loss Transformer | 提升 |
|---|---:|---:|---:|
| 50–59 | 0.145430 | 0.146153 | +0.000723 |
| 60–70 | 0.166238 | 0.166992 | +0.000754 |

全量使用 0–70 月、seed42、5 轮训练。单独提交 `joint_transformer_hybrid_epoch5_fulltrain.csv`，Kaggle submission ref `56201672`，Public LB `0.136`。旧 MSE Joint-Transformer Public LB 为 `0.130`，本次提高 `+0.006`，并超过此前 `0.135` 的项目最好提交。

结论：组合 loss 对当前 Joint-Transformer 的强化成立，当前项目最高 Public LB 更新为 `0.136`。

## 20% 完整配方混合

将 80% Public 0.135 完整配方与 20% hybrid Transformer 做单位范数混合；本地 50–59 提高 +0.000731，60–70 提高 +0.000938。Kaggle submission ref 56202423，Public LB 0.137，成为新的项目最好成绩。
