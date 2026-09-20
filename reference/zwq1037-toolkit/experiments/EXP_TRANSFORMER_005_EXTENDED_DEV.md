# EXP-TRANSFORMER-005：延长训练后的当前最佳配方检验

日期：2026-09-13  
状态：开发折完成；停止，不进入确认折。

## 目的

把 Joint-Transformer 从 6 轮继续训练到 12 轮，并在第 8、10、12 轮保存预测，检验较长训练是否能提高它在当前 Public 0.135 配方中的增量价值。

开发切分固定为训练 0–49 月、验证 50–59 月。第 7–12 轮学习率固定为 `1e-4`，其余模型结构、seed42、输入和预处理保持不变。

当前配方在本地按实际提交结构重建：67.5% Relative319 TabM trim1、18% 旧 XGB053R、4.5% 双 seed GRU embedding tree、10% 双 seed Joint-GRU。候选只把最后 10% 改成 5% Joint-GRU + 5% Joint-Transformer。

## 结果

| Transformer epoch | 单模 cosine | 完整融合 cosine | 相对当前配方 |
|---:|---:|---:|---:|
| 6 | 0.145430 | 0.155326 | -0.000114 |
| 8 | 0.143029 | 0.155337 | -0.000102 |
| 10 | 0.140734 | 0.155358 | -0.000081 |
| 12 | 0.136095 | 0.155315 | -0.000125 |

当前配方本身为 0.155440。四个检查点全部下降；第 10 轮损失最低且融合下降最少，但仍未达到预先设定的 `+0.0003` 开发门槛。单模分数从第 6 轮之后持续下降，说明继续降低训练损失没有带来更强的时间外泛化。

## 决策

停止这一版 Transformer，不运行 60–70 月确认折，也不搜索 epoch 或融合权重。下一项测试改为 Conv-GRU：只在现有 Joint-GRU 前加入 kernel=5 的局部卷积残差，以检验短期事件形状能否提供新的互补信息。

## 产物

- `scripts/exp_transformer_005_continue_dev_to12.py`
- `scripts/analyze_transformer_extended_dev.py`
- `data/interim/sequence_experiments/EXP-TRANSFORMER-001-JOINT-DEV/joint_transformer/checkpoint.pt`
- `data/interim/sequence_experiments/EXP-TRANSFORMER-005-EXTENDED-DEV/result.json`
- `data/interim/sequence_experiments/EXP-TRANSFORMER-005-EXTENDED-DEV/summary.csv`

