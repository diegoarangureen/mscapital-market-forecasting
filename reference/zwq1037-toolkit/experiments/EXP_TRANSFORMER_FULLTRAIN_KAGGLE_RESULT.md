# Joint-Transformer 单模：Kaggle 结果

Date: 2026-09-12

- Submission ref: `56188495`
- File: `joint_transformer_fulltrain.csv`
- Model: standalone Joint-Transformer, seed 42, epoch 6
- Public score: **0.130**
- Current best: **0.135**
- Difference from current best: **-0.005**
- Status: complete
- Daily submissions remaining after upload: **0**

Joint-Transformer 的 Public 单模结果低于当前 GRU 增量融合方案，但明显高于此前纯树单模提交。这个结果与本地验证一致：Transformer 已经是有竞争力的单模和有效的少量融合来源，但当前结构仍不足以替代 Relative319 TabM、XGB053R、Joint-GRU 和 GRU 嵌入树组成的 0.135 方案。

