# EXP-TABM-024：Relative319 TabM 的 24 个月时间加权 MSE

日期：2026-09-13。状态：seed42 开发折训练中。

Astra 建议的第二支线：保留主力 Relative319 的 319 特征、TabM 架构、逐成员 MSE、seed、量化预处理、batch 2048、AdamW、学习率曲线和第 15 轮不变；唯一改动是在训练 MSE 中按月份加权。第 t 月样本的未归一权重为 2^((t - 训练末月)/24)，随后把训练集平均权重归一为 1。训练 0–49 月时权重约为 0.452–1.862，不删除旧月份。

分阶段：
1. seed42 训练 0–49、验证 50–59，相对同 seed 的均匀权重 Relative319 MSE 比较单模 trim1。
2. seed42 开发有正向且值得投入时，补 seed137 同折；仅当两 seed 都正向且平均至少 +0.0005，才进入确认。
3. 确认训练 0–59，核心验证 62–70；另看去掉 66 月及 67–70。固定半衰期 24 月，不扫描更多值。

入口：`scripts/exp_tabm_024_relative319_recency24.py`。结果写在 `data/interim/tree_experiments/EXP-TABM-024-RECENCY24-SEED*-*/`。
