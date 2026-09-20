# EXP-TREE-003/004：LightGBM 单参数检查

共同基线为 EXP-TREE-002（0.1053585834）。两项实验都保留62个特征、月份0～59→60～70、训练target去均值、种子42、学习率0.03与cosine early stopping；每次只改一个参数。

| 实验 | 唯一改动 | 最佳轮数 | 总体cosine | 相对 EXP-TREE-002 | 结论 |
|---|---|---:|---:|---:|---|
| EXP-TREE-003 | `num_leaves: 31 → 63` | 273 | 0.1035076495 | -0.0018509339 | 更大容量退步，不保留 |
| EXP-TREE-004 | `min_child_samples: 100 → 200` | 354 | 0.1047429497 | -0.0006156337 | 更强正则化也退步，不保留 |

每项的 `config.json` 与 `monthly_cosine.csv` 在相应的 `data/interim/tree_experiments/EXP-TREE-003/`、`EXP-TREE-004/`；模型文本和验证预测也已单独保存。

结论只针对当前固定时间切分和62个特征：现有 LightGBM 参数不应因“更大树”或“更大叶子”而替换。 
