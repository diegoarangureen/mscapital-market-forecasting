# EXP-TREE-005：HistGradientBoosting 对照

## 设置

- 模型：`sklearn.ensemble.HistGradientBoostingRegressor`。
- 特征与切分：EXP-002 的62个特征；训练0～59月、验证60～70月。
- target：复用 EXP-TREE-002 的训练期去均值，不使用验证标签调整预测。
- 固定预算：400轮、`learning_rate=0.03`、`max_leaf_nodes=31`、`min_samples_leaf=100`、`l2_regularization=1.0`。
- `early_stopping=False`：禁止 sklearn 额外从训练期随机抽验证集；400轮在运行前固定，不从60～70月选择轮数。

## 结果

- 训练耗时：21.31秒。
- 总体 forward-validation cosine：**0.1061823145**。
- 相对 EXP-TREE-002：**+0.0008237311**。
- 相对 EXP-002：**+0.0023122753**。

完整配置、逐月cosine在 `data/interim/tree_experiments/EXP-TREE-005/`；模型与验证预测分别为 `outputs/models/hist_gradient_boosting_target_centered.joblib` 和 `outputs/predictions/hist_gradient_boosting_target_centered_valid.feather`。

## 结论

在当前受限的传统模型搜索中，HistGradientBoosting 是最好的单模型。该提升很小，不能据此断言它对未来 Kaggle 测试期一定更优；它只是在固定 forward validation 上胜出。
