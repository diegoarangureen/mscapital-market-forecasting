# Strict forward backtest: train 0-49, validate 50-59

Date: 2026-09-12

## Purpose

Investigate the large gap between the original 60-70 local validation and the
Kaggle Public LB. The formal split is strictly non-overlapping: train on months
0-49 and validate on months 50-59.

For TabM epoch selection, a nested split is used:

- inner train: months 0-39
- inner validation: months 40-49
- formal train: months 0-49
- formal validation: months 50-59

Correlation pruning is also recomputed inside each training fold. The original
0-59 correlation-pruning list is not reused because that would include the
50-59 validation period.

## B001 result

| Candidate | Overall | 50-54 | 55-59 | Monthly std | Worst | Q25 |
|---|---:|---:|---:|---:|---:|---:|
| TabM mean | 0.145682 | 0.145416 | 0.146190 | 0.013589 | 0.120488 | 0.133299 |
| XGBoost | 0.142267 | 0.142416 | 0.142619 | 0.013950 | 0.121707 | 0.133751 |
| **B001 mean (75/25)** | **0.149233** | **0.149423** | **0.149144** | 0.013763 | 0.124388 | 0.137096 |

TabM selected epoch 10 using only months 0-39 / 40-49. B001's original
60-70 score was about 0.175719, so the strict 50-59 score is lower by about
0.02649 and is materially closer to its Public LB score of 0.131.

## Corrected six-source result

The first diagnostic run used XGBoost's NumPy feature keys (`f0`, `f1`, ...)
without mapping them back to real feature names. That made the keeper ranking
alphabetical. EXP-BACKTEST-003 corrects the mapping and is authoritative.

Both the 0-39 and 0-49 correlation selections retain 289 of 307 features.
The corr-pruned TabM and cosine-loss TabM both select epoch 10.

| Candidate | Overall | 50-54 | 55-59 | Monthly std | Worst | Q25 | Change vs B001 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B001 mean** | **0.149233** | **0.149423** | 0.149144 | 0.013763 | 0.124388 | 0.137096 | 0 |
| Maximin both TabM means | 0.148946 | 0.148473 | **0.149914** | **0.013308** | **0.125035** | **0.138180** | -0.000287 |
| B005 exact | 0.148922 | 0.148461 | 0.149877 | 0.013326 | 0.125152 | 0.138146 | -0.000311 |
| Maximin exact | 0.148895 | 0.148425 | 0.149861 | 0.013343 | 0.125080 | 0.138115 | -0.000338 |

The exact Maximin improves 7 of 10 validation months versus B001, but loses
about 0.00431 in month 50 and 0.00172 in month 58. Those losses outweigh its
smaller monthly standard deviation and stronger worst-month/Q25 values.

This matches the Public LB ordering: B001 scored 0.131 and Maximin scored
0.130. Therefore 50-59 is useful as an additional generalization gate and
would have rejected the locally over-selected Maximin submission.

## Component scores on 50-59

| Component | Overall | Monthly std | Worst | Q25 |
|---|---:|---:|---:|---:|
| Original TabM mean, centered | 0.145557 | 0.013231 | 0.121161 | 0.133236 |
| Corr-pruned TabM mean | 0.146045 | 0.013754 | 0.120849 | 0.134137 |
| Cosine TabM mean | **0.146340** | 0.016379 | 0.117020 | 0.134896 |
| XGBoost | 0.142031 | 0.013039 | 0.123106 | 0.134152 |
| LightGBM | 0.136932 | **0.010040** | 0.122891 | 0.130170 |
| HistGradientBoosting | 0.093615 | 0.012838 | 0.075065 | 0.089351 |

## Decision

- Keep B001 as the best submission-backed baseline.
- Use both 50-59 and 60-70 (with the existing no-66 / 67-70 views) for future
  selection. A candidate should not be promoted from a tiny gain on only one
  period.
- Treat monthly standard deviation, worst month, and Q25 as robustness
  diagnostics, not replacements for overall cosine.
- Do not prefer the current Maximin or B005 weights over B001.

## Artifacts

- `scripts/backtest_b001_train049_valid5059.py`
- `scripts/backtest_maximin_train049_valid5059.py` (superseded diagnostic run)
- `scripts/backtest_maximin_train049_valid5059_v2.py` (authoritative correction)
- `data/interim/tree_experiments/EXP-BACKTEST-001-B001-TRAIN049-VALID5059/`
- `data/interim/tree_experiments/EXP-BACKTEST-003-MAXIMIN-GAINRANKED-TRAIN049-VALID5059/`
