# EXP-AUDIT-001 and EXP-TABM-011: drift audit and quantile preprocessing

Date: 2026-09-12

## Learning-rate control

The submitted B001 TabM uses a constant learning rate of 0.002. XGBoost uses
a constant shrinkage rate of 0.03. Only the separate cosine-loss TabM uses a
smooth cosine decay from 0.001 to 0.00001. EXP-TABM-011 keeps B001's constant
0.002 learning rate, so preprocessing is the only primary variable changed.

## EXP-AUDIT-001 feature drift

The exact deployed 307-feature schema was audited using 2,000 fixed samples
per labelled month and 100,000 test samples. Feature tables were read one at a
time with two CPU threads.

- No alignment or schema failure was found.
- Average test `abs(z) > 10` rates are below 0.3% for every feature group.
  Therefore widespread clipping is not a plausible explanation for the local
  versus Public LB gap.
- The largest group-level median PSI values are transaction gaps (0.0493),
  last-60 market features (0.0284), and full-window market features (0.0248).
- Test event coverage differs: selected transaction/order/public features have
  missing-rate reductions of roughly 5-7 percentage points.
- Trade-count and trade-volume IQRs are roughly 1.4 times their labelled-period
  values for several features, consistent with a more active test composition.

High-drift features are diagnostic only; they are not automatically removed.

## EXP-TABM-011 empirical normal-quantile preprocessing

For each training fold and feature, 1,001 empirical quantile knots are fitted
from at most 200,000 training rows. Values are mapped through the empirical CDF
to a normal score. Missing values use the training median and the existing
binary missing indicators. No validation or test statistics enter the fit.

All other TabM settings remain unchanged: K=16, two blocks, width 256, dropout
0.1, batch size 2048, AdamW, learning rate 0.002 constant, weight decay 0.0003,
and seed 42.

### TabM-only results

| Forward window | Standard TabM | Quantile mean | Change | Quantile trim1 | Change |
|---|---:|---:|---:|---:|---:|
| 0-49 -> 50-59 | 0.145682 | 0.146747 | +0.001065 | **0.146767** | **+0.001085** |
| 0-59 -> 60-70 | 0.176059 | 0.176915 | +0.000856 | **0.177073** | **+0.001014** |

The selected quantile epochs are 8 and 11 respectively.

### Frozen B001 75/25 replacement

| Forward window | Standard B001 | Quantile mean + XGB | Change | Quantile trim1 + XGB | Change |
|---|---:|---:|---:|---:|---:|
| 0-49 -> 50-59 | 0.149233 | 0.150685 | +0.001452 | **0.150699** | **+0.001466** |
| 0-59 -> 60-70 | 0.175719 | 0.176208 | +0.000489 | **0.176315** | **+0.000596** |

For 60-70, the quantile-trim1 B001 also improves:

- without month 66: 0.152220 -> 0.155262
- months 67-70: 0.150606 -> 0.155141
- monthly worst: 0.125559 -> 0.135478
- monthly Q25: 0.147015 -> 0.149501

## Current decision

Quantile preprocessing is provisionally effective because its direction agrees
on both forward windows and it improves robustness diagnostics. Before training
a full submission, add the earlier nested forward window 0-39 -> 40-49, with
0-29 -> 30-39 used only for epoch selection.

## Artifacts

- `scripts/audit_exp053r_feature_drift.py`
- `scripts/exp_tabm_011_quantile_preprocessing.py`
- `scripts/evaluate_quantile_tabm_b001_blend.py`
- `data/interim/tree_experiments/EXP-AUDIT-001-EXP053R-DRIFT/`
- `data/interim/tree_experiments/EXP-TABM-011-QUANTILE/`
- `data/interim/tree_experiments/EXP-BLEND-015-QUANTILE-B001/`
