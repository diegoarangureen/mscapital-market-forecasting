# EXP060--EXP063: selective exact-book follow-ups

## Motivation

EXP059 replaced all 35 cached book features and improved monthly stability but
reduced the main cosine scores. A target-free audit ranked exact reconstructions
by their value change and EXP053R training-only feature importance. Three leading
features were tested independently, followed by one complementary residual test.

## Fixed protocol

- Baseline: EXP053R
- Train months: 0--59
- Validation months: 60--70
- Primary robustness slice: 62--70 excluding month 66
- Recent slice: 67--70
- Model parameters: unchanged 800-tree XGBoost on CUDA

## Results

| Experiment | Change | Overall | 62--70 no66 | 67--70 | Primary std | Primary worst | Primary LOMO min | Core | Strict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| EXP053R | Baseline | 0.150847 | 0.136434 | 0.134155 | 0.011608 | 0.113105 | +0.000016 | Yes | Yes |
| EXP060 | Replace only `ofi_1_last`; cached fallback on missing | 0.148253 | 0.134372 | 0.130500 | 0.012610 | 0.110692 | -0.002729 | No | No |
| EXP061 | Replace only `microprice_displacement_last`; cached fallback on missing | 0.150412 | 0.134163 | 0.130797 | 0.012708 | 0.108951 | -0.002810 | No | No |
| EXP062 | Replace only `book_imbalance_2_std_robust`; cached fallback on missing | 0.149810 | 0.136128 | 0.134808 | 0.010443 | 0.114543 | -0.000652 | No | No |
| EXP063 | Keep cached L2 imbalance std; add exact value and exact-minus-cached residual | 0.150931 | 0.136442 | 0.133403 | 0.011892 | 0.112942 | -0.000259 | No | No |

## Decision

- Retain EXP053R as the single-model baseline.
- Reject EXP060 and EXP061 across all robustness views.
- Do not adopt EXP062: recent and stability metrics improve, but overall, no66,
  and leave-one-month-out metrics decline.
- Do not adopt EXP063: its +0.000084 overall and +0.000008 no66 changes are too
  small, while recent falls by 0.000752, primary Q25 falls by 0.002937, and LOMO
  becomes negative.
- End the exact-book replacement branch. Do not build its test features or submit it.

## Interpretation

The tree feature space is not globally exhausted, but direct additions and minor
recalculations of familiar order-book statistics are showing strong diminishing
returns. Future feature work should emphasize robust feature selection, genuinely
new representations, or raw-sequence models instead of more variants of the same
aggregates.

