# EXP-TREE-006：RandomForest 对照

## 设置

- 模型：`RandomForestRegressor`，128棵树，`max_depth=16`、`min_samples_leaf=100`、全特征候选、bootstrap采样63.2%、随机种子42。
- 特征：EXP-002 的62个特征。
- 时间划分：训练月份0～59（1,064,163行），验证月份60～70（193,474行）。
- target：仅使用训练期均值进行去均值，与 EXP-TREE-002/005 一致；没有对验证预测作目标知情校准。

## 结果

- 训练耗时：991.10秒。
- 总体 forward-validation cosine：**0.0955312218**。
- 相对当前 HistGradientBoosting 最优（0.1061823145）：**-0.0106510927**。

完整配置和逐月分数在 `data/interim/tree_experiments/EXP-TREE-006/`；模型和验证预测在 `outputs/models/random_forest_target_centered.joblib` 与 `outputs/predictions/random_forest_target_centered_valid.feather`。

## 结论

这个有界随机森林同时更慢且明显更弱，不作为后续默认模型。这里的结论限于当前62个聚合特征和固定forward validation，不推出“任何随机森林在金融任务都无效”。
