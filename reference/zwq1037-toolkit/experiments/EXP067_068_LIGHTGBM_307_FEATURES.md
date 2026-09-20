# EXP067--EXP068: LightGBM on the 307-feature EXP053R schema

## Fixed protocol

- Training months: 0--59
- Validation months: 60--70
- Primary robustness slice: 62--70 excluding month 66
- Recent slice: 67--70
- Features: the unchanged 307-column EXP053R schema
- Target: training-target centered
- LightGBM: 31 leaves, learning rate 0.03, minimum leaf size 100,
  L2=1, column fraction 0.8, two CPU threads

## Results

| Trees | Overall | 62--70 no66 | 67--70 | 60--64 | Primary std | Primary worst | Primary q25 | Primary LOMO min | Core | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| EXP053R XGBoost | 0.150847 | 0.136434 | 0.134155 | 0.141954 | 0.011608 | 0.113105 | 0.135508 | +0.000016 | Yes | Yes |
| 200 | 0.146902 | 0.131498 | 0.130570 | 0.133827 | 0.009072 | 0.121074 | 0.126188 | -0.007228 | No | No |
| 400 | 0.151208 | 0.134832 | 0.133720 | 0.138920 | 0.009418 | 0.123804 | 0.130277 | -0.003486 | No | No |
| 600 | 0.151612 | 0.135248 | 0.132504 | 0.140789 | 0.010557 | 0.123257 | 0.129411 | -0.002921 | No | No |
| 800 (EXP067) | **0.152150** | 0.136043 | 0.133410 | **0.141582** | 0.010634 | 0.123168 | 0.129326 | -0.001987 | No | No |
| 900 prefix | 0.152104 | 0.136344 | 0.133954 | 0.141128 | **0.010330** | 0.122560 | **0.129601** | -0.001542 | No | No |
| 1000 (EXP068) | 0.151585 | **0.136625** | **0.134187** | 0.141039 | 0.010341 | **0.122720** | 0.129600 | **-0.001239** | No | No |

## Decisions

- Keep EXP053R XGBoost as the robust single-model baseline.
- Do not submit EXP067 or EXP068 as a replacement. LightGBM improves overall and
  worst-month scores, but every checked tree count has a negative primary LOMO and
  a materially lower primary q25 than EXP053R.
- Stop the LightGBM tree-count search. The 800--1000 region was checked densely
  enough to show the tradeoff rather than a missing optimum.
- Earlier 62-feature tests already found that increasing leaves to 63 or minimum
  leaf size to 200 reduced validation cosine, so no additional one-parameter retry
  is justified on the current validation budget.
- Preserve EXP067/068 only as potential diversity candidates if model blending is
  reconsidered later.

## Overall judgment

Conventional hand-crafted tree features are at a local saturation point under the
current strict panel. The evidence now includes failed additive features, exact
reconstructions, selective replacements, low-gain pruning, correlation pruning,
state interactions, and a full 307-feature LightGBM comparison. Further small tree
feature or parameter variants are more likely to overfit months 60--70 than deliver
a robust gain.

