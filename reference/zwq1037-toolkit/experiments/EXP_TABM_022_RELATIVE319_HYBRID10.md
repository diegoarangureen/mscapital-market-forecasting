# EXP-TABM-022：Relative319 TabM 的 90% MSE + 10% cosine loss

状态：seed42 开发与确认折、seed137 开发折完成；不采用（2026-09-13）。

唯一主改动：在主力 319 特征 TabM 的逐成员 MSE 中加入 10% 的逐成员 batch cosine loss；保持特征、seed、架构、量化预处理、batch 2048、AdamW、学习率曲线和 15 轮不变。

| 验证窗口 | TabM mean MSE | 混合损失 | 单模增量 | 固定 B001 trim1 增量 |
|---|---:|---:|---:|---:|
| 50–59 开发折 | 0.151364 | 0.152367 | +0.001003 | +0.000658 |
| 60–70 确认折 | 0.176697 | 0.176885 | +0.000189 | +0.000285 |
| 62–70，两个月间隔 | 0.178869 | 0.179257 | +0.000387 | +0.000416 |
| 62–70，去掉 66 月 | 0.153872 | 0.153868 | -0.000005 | +0.000043 |
| 67–70 | 0.152217 | 0.153215 | +0.000999 | +0.000697 |

结论：seed42 开发折与近期月份正向，但整体确认增益较小且 66 月贡献明显；seed137 的 50–59 月开发折单模 -0.000580、固定 B001 trim1 -0.000668。增益未跨 seed 复现，不替换主力 MSE，不继续 seed137 确认折，也不提交 Kaggle。

产物：
- `scripts/exp_tabm_022_relative319_hybrid10.py`
- `scripts/analyze_tabm_022_loss.py`
- `data/interim/tree_experiments/EXP-TABM-022-RELATIVE319-HYBRID10/comparison.json`
- `data/interim/tree_experiments/EXP-TABM-022-RELATIVE319-HYBRID10/confirmation_comparison.json`

