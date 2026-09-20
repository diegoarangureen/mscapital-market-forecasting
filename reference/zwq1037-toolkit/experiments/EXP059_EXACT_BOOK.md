# EXP059: exact full-sequence book reconstruction

## Fixed protocol

- Baseline: EXP053R
- Train months: 0--59
- Validation months: 60--70
- Primary robustness slice: 62--70 excluding month 66
- Recent slice: 67--70
- Model: unchanged 307-feature XGBoost, 800 trees, CUDA

## Implementation

- Rebuilt the same 35 book microstructure columns from all 221,756,611 raw market rows.
- Used a restartable two-thread builder with 1,000 samples per checkpoint.
- Completed 1,258 checkpoints and produced 1,257,637 rows by 36 columns.
- Only feature values changed; the feature names and model parameters stayed fixed.

## Result

| Metric | EXP053R | EXP059 | Delta |
|---|---:|---:|---:|
| Overall cosine | 0.150847 | 0.149938 | -0.000909 |
| 62--70 excluding 66 | 0.136434 | 0.135540 | -0.000894 |
| 67--70 | 0.134155 | 0.133149 | -0.001005 |
| Primary monthly std | 0.011608 | 0.010298 | -0.001310 |
| Primary worst month | 0.113105 | 0.116479 | +0.003373 |
| Primary LOMO minimum change | +0.000016 | -0.001557 | -0.001573 |

- Improved validation months: 4 of 11
- Declined validation months: 7 of 11
- Core generalization gate: failed
- Strict stability gate: failed

## Decision

Do not replace EXP053R with EXP059 and do not build an EXP059 submission. The exact
book reconstruction is more stable by monthly standard deviation and worst-month
score, but it is weaker on overall, no66, recent, and leave-one-month-out metrics.

