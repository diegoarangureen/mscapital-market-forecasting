# EXP-TREE-002：LightGBM 训练 target 去均值

## 假设

EXP-002 的 L2 预测含有常数方向成分。只在训练月份0～59计算均值 $\bar y_{train}$，训练目标改为 $y-\bar y_{train}$，预测时不加回该均值，可能使预测方向更贴近官方未中心化 cosine。

## 公平设置

- 特征：与 EXP-002 完全相同的62个特征。
- 划分：训练0～59月（1,064,163行），验证60～70月（193,474行）。
- 保持：`learning_rate=0.03`、`num_leaves=31`、`min_child_samples=100`、`reg_lambda=1.0`、随机种子42、5000轮上限、验证cosine patience 200。
- 唯一改动：训练目标从 `target` 改为 `target - (-2.1809362806379795e-05)`；验证target不参与这个变换。

## 结果

- 最佳轮数：411。
- 训练耗时：31.96秒。
- 总体 forward-validation cosine：**0.1053585834**。
- 相对 EXP-002（0.1038700392）：**+0.0014885442**。
- 验证预测均为有限值，193,474个 `sample_id` 唯一，且与 EXP-002 预测文件的ID、月份和target逐行对齐。

完整配置、模型、验证预测与逐月分数分别位于 `data/interim/tree_experiments/EXP-TREE-002/`、`outputs/models/lightgbm_market_v1_last60_target_centered.txt` 和 `outputs/predictions/lightgbm_market_v1_last60_target_centered_valid.feather`。

## 结论

训练期target去均值是当前有效且无验证标签泄漏的 cosine 对齐做法，成为后续传统树模型对照的共同 target 处理。它不是“把官方指标中心化”，而是一次经固定未来月份验证的训练目标平移。
