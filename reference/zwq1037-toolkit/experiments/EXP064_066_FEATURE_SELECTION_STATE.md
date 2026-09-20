# EXP064--EXP066: pruning and flow-state tests

## Fixed protocol

- Baseline: EXP053R with 307 features
- Train months: 0--59
- Validation months: 60--70
- Primary robustness slice: 62--70 excluding month 66
- Recent slice: 67--70
- Model parameters: unchanged 800-tree XGBoost on CUDA

## Results

| Experiment | Change | Features | Overall | 62--70 no66 | 67--70 | Primary std | Primary worst | Primary LOMO min | Core | Strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| EXP053R | Baseline | 307 | 0.150847 | 0.136434 | 0.134155 | 0.011608 | 0.113105 | +0.000016 | Yes | Yes |
| EXP064 | Drop bottom 10% by training-only total gain | 277 | 0.150050 | 0.135439 | 0.132768 | 0.011784 | 0.111195 | -0.001557 | No | No |
| EXP065 | Drop 14 pairs with training-only absolute correlation >= 0.995 | 293 | 0.150503 | 0.135614 | 0.133698 | 0.011333 | 0.112744 | -0.002067 | No | No |
| EXP066 | Add three final-20-second directional flow-alignment strengths | 310 | 0.149345 | 0.135406 | 0.133362 | 0.011678 | 0.111940 | -0.002074 | No | No |

## Decisions

- Reject EXP064. Low training gain did not imply low out-of-time value.
- Reject EXP065. Near-duplicate columns still carried useful small differences for trees.
- Reject EXP066. Explicit agreement between book imbalance, OFI, and trade imbalance did not improve generalization.
- Keep EXP053R unchanged.
- Stop expanding conventional hand-crafted tree features. The next fair test should keep the 307-feature schema fixed and change the model family.

## Interpretation

Feature engineering is not globally exhausted, but the current family of manually
aggregated order-book, order-flow, transaction, rolling-window, and interaction
features is at a local saturation point under the strict validation panel. Further
small formula variations are more likely to fit validation noise than add robust
information.

