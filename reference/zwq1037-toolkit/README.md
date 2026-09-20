# MSCapital Financial Market Forecasting Toolkit

This is the public code and experiment archive behind a **0.152 public
leaderboard** solution for the Kaggle competition
[MSCapital – Real Financial Market Forecasting](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/overview).

## What is included

- Feature engineering for market, order-flow, transaction-flow, path,
  volatility, spectral, and cross-stream signals.
- Time-aware validation with train months 0–59, purge months 60–61, and
  validation months 62–70 while excluding anomalous month 66.
- TabM, RealMLP, GRU, factorized Transformer, tree-model, TSMixer, TCN, and
  hybrid experiments.
- EMA, temporal folds, loss experiments, model screening, and constrained
  blend searches.
- Experiment reports and the final project closeout record.

## What is intentionally excluded

Competition data, downloaded third-party notebooks, trained checkpoints,
cached features, prediction files, and submissions are not redistributed.
Obtain the competition data from Kaggle and follow its rules. Some experiment
scripts retain the original directory assumptions and are best treated as a
research archive; start with `docs/`, `configs/`, and the most recent scripts.

## Final validation protocol

1. Train on months 0–59.
2. Purge months 60–61.
3. Validate on months 62–70, excluding month 66.
4. Select on months 62–65 and confirm forward stability on months 67–70.
5. Require improvement in the selection, forward, and combined windows before
   full training or submission.

## Main finding

The strongest final blend combined public TabM/YangQ predictions with owned
TabM, RealMLP, GRU, and a Transformer slot. The Transformer slot mixed a raw
factorized Transformer (40%) with a market-conditioned event-residual
Transformer (60%). Further blend tuning did not improve the displayed 0.152
score; the limiting factor was new single-model signal rather than blend
weights.

See `PROJECT_CLOSEOUT.md` for exact weights, accepted evidence, rejected
directions, and reopening criteria.

## 中文说明

这是该比赛项目的精简开源归档，包含我们自己编写的特征、模型、验证、
EMA、时间折与融合搜索代码。比赛数据、他人 notebook、权重、缓存和预测
文件没有再分发。最终公开榜成绩为 **0.152**，详细实验结论见
`PROJECT_CLOSEOUT.md`。

## License

Code is released under the MIT License. Documentation and experiment reports
are released under CC BY 4.0. Third-party dependencies keep their own licenses.

## Kaggle Transformer source

The exact temporal 3-fold Factorized Transformer (standalone public LB 0.145), the event-residual Transformer used in the final blend, and their cache builder are in `kaggle_transformers/`.

