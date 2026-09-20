# MSCapital 项目收尾记录（2026-09-20）

## 最终结果

- 最佳公开榜分数：**0.152**。
- 当前推荐归档提交：`outputs/submissions/current152_transformer020raw40_event60.csv`。
- Kaggle submission ref：`56361734`。
- 同分参考：`current152_owned_softgate25_transfer.csv` 与 `current151_theoretical_geometry_optimum.csv`。
- 本轮正式提交预算：已使用 1/3，剩余 2/3；因没有新候选通过本地与前向门槛，剩余次数不再消耗。

## 最终配方

公开模型块保持固定：

- GPU TabM：0.2108933794
- TPU TabM：0.0565335992
- YangQ：0.2037419935
- 公开块合计：0.4711689721

自有模型块合计为 0.5288310279，使用基于 `realized_volatility_60` 的软门控。Transformer 槽位最终采用：

- raw Factorized Transformer：40%
- market-conditioned event-residual Transformer：60%

Transformer 槽位的平均有效总权重约 24.24%，折算到整份提交约为 raw 9.70%、event residual 14.54%；其余约 75.76% 来自公开块和 TabM、GRU、RealMLP 等保留组件。

## 固定验证协议

- 训练：月份 0–59
- purge：月份 60–61
- 验证：月份 62–70，排除异常月份 66
- 选权：月份 62–65
- 前向确认：月份 67–70
- 候选必须同时改善选择区间、前向区间和全区间，并通过逐月稳定性门槛后才允许全量训练或正式提交。

## 关键证据

raw40/event60 相对旧软门控基线的离线结果：

- 选择区间：+0.00169938
- 前向区间：+0.00250331
- 全区间：+0.00221353
- 前向 4/4 个月为正
- 选择月份 LOMO 4/4 为正
- 公榜：0.152，与原最佳显示分数持平

## 已排除的主要方向

- Patch-Transformer-GRU：全区间较 EXP020 低 0.01246。
- Axial-LOB：第三轮 EMA 全区间较 EXP020 低 0.01726，停止续跑。
- month-macro cosine loss、recent-head adaptation、multiwindow EMA、高波动专家：未通过前向或稳定性门槛。
- 树、EMA TabM、subsecond、multiwindow 的扩展融合：选择区间可升，但前向不升。
- TSMixer、线性序列、last-readout、TCN、Deep3 Transformer、Transformer EMA：最佳融合权重为 0%。
- Relative394：选权得到 15%，但前向下降 0.00043。
- 25% 月内秩目标 + 75% 原始目标 TabM：选择区间下降 0.00187，前向下降 0.00076。
- 纯月内秩目标树模型：两窗分别下降约 0.0105 和 0.0116。
- 继续细化融合权重：未找到同时通过选择、前向和逐月稳定性门槛的组合。

## 项目结论

当前瓶颈已经不是融合参数，而是缺少新的单模信号。公开模型仍提供不可替代的私有特征信息；仅依赖现有公开特征与已有模型继续微调，预期收益低于实验和提交成本。

## 未来重新开启的条件

仅在以下情况之一出现时重新开启：

1. 有新的、规则允许的公开特征构造代码，并且不是现有 OFI、订单流、路径、频谱或窗口统计的重复版本。
2. 新单模在 62–65 与 67–70 都稳定提升，并至少在三个前向月份为正。
3. 比赛论坛发布可复现的新结构或训练方法，能够解释为何与现有 Transformer、GRU、TabM 形成不同信号。
4. 需要为最终私榜重新选择已验证的同分提交。

## 主要文件

- 最终提交：`outputs/submissions/current152_transformer020raw40_event60.csv`
- 最终融合元数据：`outputs/submission_metadata/current152_transformer020raw40_event60.json`
- Kaggle 提交记录：`outputs/submission_metadata/current152_transformer020raw40_event60_kaggle_submission.json`
- 完整续跑计划：`outputs/submission_metadata/goal_improve_beyond152_20260919.json`
- 最终候选几何检查：`outputs/submission_metadata/event_residual_v28_full_candidate_geometry.json`
- 扩展融合搜索：`outputs/submission_metadata/extended_owned_blend_search_20260920.json`
- 保守融合搜索：`outputs/submission_metadata/conservative_extended_owned_blend_search_20260920.json`
- 旧序列模型筛选：`outputs/submission_metadata/previous_sequence_slot_screen_20260920.json`