# Experiment ledger

Validation = cosine on months 61-70, models trained on months 0-60.
Seed-average of 3 seeds unless noted. "Centered" = mean-centered predictions.

## Progression

| Model | Val cosine |
|---|---|
| Ridge (22 basic features) | 0.0625 |
| LightGBM baseline (58 features) | 0.1213 |
| Coordinate sweep best (bagging 0.7) | 0.1232 |
| Bagging 0.7, 3-seed average | 0.1258 |
| + mean-centered predictions (GBM v4, submitted) | **0.1274** |

GBM v4 scored 0.110 on the public leaderboard (49% of test), rank 197/206.

## Sweep (13 configs, single-seed then seed-averaged)

Best single config bagging 0.7 (0.1232). Runner-up bag07_l2_10 (0.1242)
dropped to 0.1268 centered once seed-averaged - config differences below
~0.002 are seed variance.

## Ensembles and blends

Greedy forward-selection blend over 26 saved validation prediction variants
tops out around 0.129 - above the champion but below the 0.130 bar set for
spending a submission, and partly selection bias on the validation months.

## Negative results

| Idea | Result |
|---|---|
| +8 L2 order-book features (X3) | 0.1274 - flat |
| +11 microstructure features (X4) | 0.1217 - worse |
| Tick-rule / BVC trade-sign features (X5) | 0.1216 - worse |
| Extra-trees | 0.1214 single seed - worse |
| Dart boosting | 0.1208 centered - worse |
| Per-month normalization / ranking | hurts (target has zero lag-1 autocorr) |
| Neighbor smoothing | hurts |

## Structural findings

- `transaction.side` is the true aggressor: 63% agreement with the best
  tick-rule reconstruction, IC +0.075 vs +0.027. Inferring signs adds noise.
- Bulk-volume-classification imbalance is contrarian here (IC ~ -0.005).
- Zero exact duplicate market-window signatures in a 5,000-sample audit of
  month 61: no same-window twins to exploit.
- Robustness split (train 0-50, validate 51-60): 0.1208 - validation is honest.

## What top-10 would need

The leader sits at 0.172 public vs 0.110 here. That gap does not look
tunable: it likely requires cross-asset factor structure or sequential
(temporal) models over the month axis, i.e. a different model class rather
than better features for a per-sample GBM.
