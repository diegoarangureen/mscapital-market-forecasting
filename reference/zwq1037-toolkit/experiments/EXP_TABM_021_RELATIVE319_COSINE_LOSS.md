# EXP-TABM-021：Relative319 TabM 的 batch cosine loss

状态：开发折与确认折完成；不采用。

目标：提高主力 Relative319 TabM 单模。保持 319 特征、seed 42、TabM 架构、量化预处理、batch 2048、AdamW、学习率曲线和 15 轮训练不变，只把逐成员 MSE 换为逐成员 batch cosine loss。目标的线性标准化不改变中心化 cosine；预测沿用原尺度还原与评估流程。

| 验证窗口 | TabM mean MSE | TabM mean cosine loss | 增量 | B001 trim1 增量 |
|---|---:|---:|---:|---:|
| 50–59 开发折 | 0.151364 | 0.151493 | +0.000129 | +0.000197 |
| 60–70 确认折 | 0.176697 | 0.174864 | -0.001832 | -0.001430 |
| 62–70，两个月间隔 | 0.178869 | 0.177202 | -0.001668 | -0.001363 |
| 62–70，去掉 66 月 | 0.153872 | 0.153521 | -0.000351 | -0.000144 |
| 67–70 | 0.152217 | 0.151988 | -0.000228 | -0.000106 |

开发折的小增益未跨窗口复现。尤其在剔除 66 月后，单模仍低于 MSE，因此不替换主力训练损失，不提交 Kaggle。下一步只考虑预先定义的一项更温和损失配方，避免在纯 cosine 上反复调权重。

产物：
- `scripts/exp_tabm_021_relative319_cosine_loss.py`
- `scripts/analyze_tabm_021_loss.py`
- `data/interim/tree_experiments/EXP-TABM-021-RELATIVE319-COSINE-LOSS/comparison.json`
- `data/interim/tree_experiments/EXP-TABM-021-RELATIVE319-COSINE-LOSS/confirmation_comparison.json`
