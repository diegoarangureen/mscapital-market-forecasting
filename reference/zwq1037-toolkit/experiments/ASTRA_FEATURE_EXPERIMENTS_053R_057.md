# Astra feature experiments: EXP053R--EXP057

## Fixed evaluation protocol

- Training months: 0--59
- Validation months: 60--70
- Primary robustness score: months 62--70 excluding month 66
- Recent score: months 67--70
- Stability checks: primary monthly standard deviation, worst month, and primary leave-one-month-out minimum change
- Model: unchanged EXP053 XGBoost parameters, 800 trees, CUDA

## Results

| Experiment | Change from EXP053R | Overall | 62--70 no66 | 67--70 | Primary std | Primary worst | LOMO min | Core | Strict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| EXP053 | Original cached full-RV denominator | 0.149167 | 0.135324 | 0.132709 | 0.011776 | 0.113035 | -0.000589 | No | No |
| EXP053R | Exact full raw-sequence `m_rv` denominator | **0.150847** | **0.136434** | 0.134155 | 0.011608 | 0.113105 | +0.000016 | **Yes** | **Yes** |
| EXP054 | Three 60-second imbalance persistence features | 0.149847 | 0.134376 | 0.131776 | 0.010820 | 0.113233 | -0.002410 | No | No |
| EXP055 | Two normalized CVD path features | 0.149681 | 0.136402 | **0.135062** | **0.010612** | 0.113206 | -0.000536 | No | No |
| EXP056 | Two signed-flow versus price-response interactions | 0.147535 | 0.133208 | 0.130525 | 0.010594 | **0.114765** | -0.003971 | No | No |
| EXP057 | Three stable liquidity-tail features | 0.148906 | 0.134567 | 0.132219 | 0.011041 | 0.113048 | -0.002230 | No | No |

## Decisions

- Keep EXP053R as the new single-model baseline.
- Reject EXP054, EXP056, and EXP057.
- Do not merge EXP055 despite its recent-period gain: its no66 score is slightly lower, early score falls by 0.002711, and its primary LOMO minimum is negative.
- The existing EXP053 five-fold submission already recomputes `x_rv_15_over_full` from the complete public `m_rv`, so its Public LB 0.122 corresponds to the corrected feature definition.
- Do not build test versions of the rejected feature groups.

## Leakage and implementation audit

- Imbalance persistence uses a fixed +/-0.05 dead zone and real elapsed seconds within each sample only.
- CVD paths use side 0 as buy and side 1 as sell, prepend a conceptual zero path origin, and normalize by within-sample total volume.
- Liquidity quantiles are computed independently inside each sample from the complete raw market sequence.
- No feature uses target values, validation-fitted thresholds, month IDs, cross-sample ranks, or batch-level quantiles.
