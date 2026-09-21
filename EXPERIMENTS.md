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

## Era 2: RealMLP (Sep 15-21)

The GBM line above was superseded by an ensemble-MLP (RealMLP: PBLD periodic
embeddings, NTP-linear members, EMA, weighted-MSE + cosine loss, label-noise
regularization) on a 450-column feature matrix (298 proprietary + 152 public
rfmf-0726). Full numbers in research/LOG.md; headline ledger:

| Experiment | Change | OOF/val | LB | Verdict |
|---|---|---|---|---|
| v9 | RealMLP 246f, refit-on-full | - | 0.124 | baseline |
| v13 | 450f, 5 purged folds x 3 seeds, holdout+ES, 15-model mean | 0.1671 | **0.139** | champion |
| v14 | v13 + 75 cross-sectional rank features | +OOF | 0.135 | XS features LB-toxic |
| v15 | 5 folds x 5 seeds (25 models) | 0.1683 | 0.138 | aggregation ceiling |
| X21-X22 | order-flow stream aggregates | +0.002 val | - | folded into 450f |
| X23/X24c/X25/X26 | 4 feature families, each controlled A/B | flat (<=+0.001) | - | feature engineering closed |
| RS1/RS2 | random hyperparameter search, 469f | 0.1666 / 0.1579 | - | champion locally optimal |
| TabM@450f | bestwater architecture, reimplemented (Apache 2.0), folds 4-5 | 0.1521 | - | his edge = 462 private features, not the arch |
| TabM blend | z-scored RealMLP+TabM blend, forward-sim | +0.001 | - | below LB resolution, no case |
| X28a | loss balance lambda_cos 1.0 -> 0.1 | 0.1683 vs 0.1701 (fold5) | - | loss-balance closed |
| X28b | + RQ-KMeans auxiliary target-code head | queued | - | next lever (untested by anyone public) |

Cross-program evidence (a second, independent 130-experiment public toolkit,
mirrored in reference/): identical walls - local feature gains fail to
transfer to the leaderboard (3 independent observations), tree features
saturate past ~300 columns, sequence models yield +0.001 forward deltas at
best, and its "improve beyond 0.152" campaign closed with no verified gain.
Public-data approaches plateau at 0.15-0.16.

## What top-10 needs (updated Sep 21)

Top-10 cut is 0.161; leader 0.176. With public features alone the realistic
ceiling is ~0.15-0.16 (two independent programs converge on this). The
remaining levers, in order: (1) auxiliary-target training (RQ head, X28b);
(2) TPU-scale seed/fold replication for robustness (pending account
verification); (3) proprietary order-flow signal mining beyond the public
152, gated by strict forward simulations (the known trap: +0.001 local can
be -0.015 on the leaderboard).
