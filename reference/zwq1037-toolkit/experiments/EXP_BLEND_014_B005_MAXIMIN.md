# EXP-BLEND-014: B005-to-robust maximin interpolation

Date: 2026-09-11

## Purpose

Reduce the two remaining primary-month regressions of the double-trim B005 blend
without adopting a high-dimensional validation-selected weight vector.

## Method

Search only the one-dimensional line between:

- Double-trim B005 weights: 0.35 original TabM, 0.20 corr-pruned TabM,
  0.15 cosine TabM, 0.10 XGBoost, 0.10 LightGBM, 0.10 HistGB.
- Robust-grid weights: 0.40, 0.20, 0.20, 0.05, 0.05, 0.10 in the same order.

Alpha was checked from 0.00 to 1.00 in increments of 0.01. Selection maximized
the minimum per-month cosine change versus public-best B001 over months 62--70,
excluding month 66.

## Result

The maximin point is alpha 0.05, giving weights:

- Original TabM trim1: 0.3525
- Corr-pruned TabM mean: 0.2000
- Cosine TabM trim1: 0.1525
- XGBoost: 0.0975
- LightGBM: 0.0975
- HistGB: 0.1000

| Candidate | Overall | 62--70 no66 | 67--70 | 60--64 | Primary std | Worst | q25 | Worst month change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Double-trim B005 | 0.175939 | 0.154237 | 0.153217 | 0.155329 | 0.012937 | 0.132615 | 0.147766 | -0.001314 |
| **Alpha 0.05 maximin** | **0.175986** | **0.154264** | **0.153232** | **0.155362** | 0.012948 | **0.132631** | 0.147750 | **-0.001248** |

At alpha 0.05 the two remaining regressions are almost equal: month 65 changes
by -0.00124782 and month 69 by -0.00124667 versus B001. This is a materially more
balanced reason for the small weight change than maximizing aggregate validation
score. Primary LOMO minimum change is +0.001469.

On test, this candidate is correlated 0.993833 with B001 and differs by 11.56% in
relative L2 direction. It is correlated 0.999996 with double-trim B005 and differs
from it by only 0.274%, consistent with the intentionally small interpolation.

Prepared submission:
`outputs/submissions/tabm_double_trim_b005_maximin_alpha005_fulltrain.csv`

Status: prepared and validated, not uploaded.

