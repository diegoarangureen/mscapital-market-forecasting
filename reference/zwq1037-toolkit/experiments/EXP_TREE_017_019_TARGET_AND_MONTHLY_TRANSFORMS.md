# EXP-TREE-017 至 EXP-TREE-019：目标处理与同月横截面变换

## 实验目的

从一份约 0.142 的 TabM 方案中，仅迁移适合树模型且能独立消融的想法：原始 target、同月 percentile rank 和同月 z-score。没有采用其中心化 cosine loss、全量标签监督选特征、跨折未来标准化或样本内异质特征排名。

固定基线为 EXP-TREE-016：132 个特征、XGBoost 深度 5、400 棵树、训练 target 减去训练期均值。正式口径始终为月份 0--59 训练、60--70 验证，以及官方未中心化 whole-vector cosine。

## 防泄漏规则

- 内部折：0--49 训练、50--59 验证。
- 每个折用该折训练区间单独训练 XGBoost importance selector，只选择 gain 最高的 12 个已有特征。
- 验证标签从未用于特征选择、rank 或 z-score。
- rank/z-score 使用同月全部无标签特征行，属于 transductive preprocessing。
- 本地测试 `market.feather` 不含 `month` 字段，因此当前无法在正式测试集原样复现同月变换。

## 结果

| 实验 | 唯一主要改动 | 特征数 | 内部 cosine | 正式 cosine | 正式变化 | 逐月标准差 | 逐月极差 | 改善月份 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EXP-TREE-016（基线） | target centered | 132 | 0.11538505 | 0.12335253 | -- | 0.01585484 | 0.05475717 | -- |
| EXP-TREE-017 | 改用原始 target | 132 | 0.11582401 | 0.12207729 | -0.00127524 | 0.01705795 | 0.06226509 | 4/11 |
| EXP-TREE-018 | 为训练折 top-12 增加同月 rank | 144 | 0.11517186 | **0.12815762** | **+0.00480509** | 0.01865272 | 0.07005079 | 8/11 |
| EXP-TREE-019 | 为训练折 top-12 增加同月 z-score | 144 | 0.11852665 | 0.10992876 | -0.01342377 | 0.02052895 | 0.08336315 | 6/11 |

## 正式 rank 特征使用的 12 个原始列

1. `last60_ask_price_1_end`
2. `ofi_1_20_mean`
3. `bid_price_1_last`
4. `ofi_1_last`
5. `last60_ask_price_2_end`
6. `microprice_displacement_last`
7. `microprice_displacement_20_mean`
8. `ask_volume_1_sum`
9. `last60_bid_volume_2_mean`
10. `ofi_1_60_mean`
11. `seconds_coverage`
12. `last60_bid_price_2_start`

## 结论

1. 原始 target 内部略升但正式下降，且月度波动增大。继续保留预测与模型产物作对照，主线仍采用训练均值中心化 target。
2. percentile rank 正式分数提高到 0.12816，但内部折未提升，逐月波动与极差也明显变大；它不是跨时期一致的增益。
3. z-score 内部折提升，正式验证却大幅下降，尤其月份 66 和 68 分别下降约 0.04475 和 0.03702，说明均值/标准差对市场状态变化非常敏感，不保留。
4. rank 依赖验证月份边界，而测试数据没有 `month` 列。因此 0.12816 只能视为研究性 transductive 结果，当前可复现到测试流程的最佳树模型仍是 EXP-TREE-016（0.12335）。
5. 暂不做 3-seed ensemble：rank 没有在内部折复现且当前不可部署，尚不满足“单 seed 真实稳定增益”的前提。

## 后续建议

若继续这个方向，应独立测试“历史参考分布”版本：仅用当前训练月份拟合每个选中列的经验分布或稳健位置/尺度，再固定映射到未来月份。它不需要知道测试月边界，能够区分 rank 思路本身与 transductive 月份信息带来的增益。

## 产物

- 运行脚本：`scripts/exp_tree_017_019_xgboost_target_and_monthly_transforms.py`
- EXP-TREE-017 至 019 配置：`data/interim/tree_experiments/<experiment_id>/config.json`
- rank 验证预测：`outputs/predictions/xgboost_microstructure_monthly_rank_valid.feather`
- z-score 验证预测：`outputs/predictions/xgboost_microstructure_monthly_zscore_valid.feather`
- 原始 target 验证预测：`outputs/predictions/xgboost_microstructure_raw_target_valid.feather`
