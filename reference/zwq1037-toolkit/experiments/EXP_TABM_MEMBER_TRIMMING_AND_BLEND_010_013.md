# TabM member trimming and robust blends (EXP-BLEND-010--013)

Date: 2026-09-11

## Evaluation protocol

- Fixed validation months: 60--70.
- Primary score: cosine on months 62--70 excluding month 66.
- Secondary slices: months 67--70 and 60--64.
- Stability diagnostics: primary monthly standard deviation, worst month, q25,
  per-month changes, and primary leave-one-month-out (LOMO) minimum change.
- Month 66 is reported but deliberately excluded from primary model selection.
- All member aggregation methods below are label-free at prediction time.

## Label-free TabM member aggregation

Each TabM model contains 16 parallel members. `trim1` sorts the 16 predictions for
each row, removes the lowest and highest values, and averages the remaining 14.

| Model | Aggregation | Overall | 62--70 no66 | 67--70 | 60--64 | Primary std | Worst | q25 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original MSE TabM | mean | 0.176059 | 0.151066 | 0.149038 | 0.152685 | 0.014201 | 0.124334 | 0.145376 |
| Original MSE TabM | trim1 | **0.176211** | **0.151312** | **0.149293** | **0.152812** | **0.014160** | **0.124604** | **0.145752** |
| Cosine-loss TabM | mean | 0.170226 | 0.152865 | 0.149993 | 0.156928 | 0.011998 | 0.136677 | 0.144200 |
| Cosine-loss TabM | trim1 | **0.170385** | **0.152980** | **0.150139** | **0.156994** | 0.012122 | **0.136709** | **0.144228** |
| Corr-pruned TabM | mean | 0.175090 | 0.151445 | 0.149221 | **0.152974** | **0.013907** | 0.125889 | **0.144906** |
| Corr-pruned TabM | trim1 | **0.175151** | **0.151497** | **0.149315** | 0.152634 | 0.014047 | **0.125892** | 0.144795 |

Decision: use `trim1` for the original and cosine-loss TabM, but retain the normal
mean for the corr-pruned TabM. The latter's small primary gain does not compensate
for its lower early score, larger monthly variation, and lower q25.

## Four-source result

The double-trim four-source candidate uses:

- 45% centered original TabM trim1
- 20% corr-pruned TabM mean
- 20% cosine-loss TabM trim1
- 15% centered XGBoost EXP053R

It scores overall 0.176427, no66 0.153808, recent 0.152129, early 0.155325,
primary std 0.013168, worst 0.130407, and q25 0.147221. Its primary LOMO
minimum improvement over the corresponding mean baseline is +0.000091.

Prepared file:
`outputs/submissions/tabm_double_trim_four_source_aggressive_45_20_20_xgb15_fulltrain.csv`

## Six-source search and anti-overfitting decision

The six sources are original TabM trim1, corr-pruned TabM mean, cosine TabM trim1,
XGBoost, LightGBM EXP068, and HistGB EXP007. Original TabM, XGBoost, and LightGBM
are globally centered before per-source L2 normalization; the other sources are
only L2-normalized.

A predeclared small grid evaluated 297 combinations. The highest no66 candidate
reached 0.154554, but it was selected from the same validation months and moved
more loss into month 69. It is retained as an aggressive research candidate, not
the default final submission.

The preferred candidate keeps the already selected B005 weights and changes only
the two member aggregation rules:

- 35% original TabM trim1
- 20% corr-pruned TabM mean
- 15% cosine-loss TabM trim1
- 10% XGBoost
- 10% LightGBM
- 10% HistGB

| Candidate | Overall | 62--70 no66 | 67--70 | 60--64 | Primary std | Worst | q25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public-best B001 local | 0.175719 | 0.152004 | 0.150606 | 0.153459 | 0.013639 | 0.125559 | 0.146702 |
| Double-trim four-source | 0.176427 | 0.153808 | 0.152129 | 0.155325 | 0.013168 | 0.130407 | 0.147221 |
| **Double-trim B005 weights** | 0.175939 | **0.154237** | **0.153217** | **0.155329** | **0.012937** | **0.132615** | **0.147766** |
| Grid highest no66 | 0.176605 | 0.154554 | 0.153259 | 0.155816 | 0.013150 | 0.132731 | 0.147233 |

Against B001, double-trim B005 improves six of the eight primary months. Its two
declines are month 65 (-0.001314) and month 69 (-0.001213). Month 66 declines by
-0.002604 but is excluded by protocol. The grid-highest candidate reduces the
month-65 decline to -0.000187 but increases the month-69 decline to -0.002125,
which is a less balanced validation tradeoff.

The six-source validation-to-test prediction geometry is stable enough for use as
a diagnostic: mean absolute pairwise correlation shift is 0.01297 and maximum is
0.03060. The preferred B005 candidate differs from public B001 by 12.73% in
validation and 11.60% on test, with correlations 0.99251 and 0.99379 respectively.

Preferred prepared file:
`outputs/submissions/tabm_double_trim_blend005_fulltrain.csv`

## Fulltrain artifacts

- Original TabM member aggregations:
  `outputs/predictions/tabm001_member_aggregations_fulltrain_test.feather`
- Cosine TabM member aggregations:
  `outputs/predictions/tabm003_cosine_member_aggregations_fulltrain_test.feather`
- Fulltrain LightGBM EXP068 prediction:
  `outputs/predictions/lightgbm068_fulltrain_test.feather`
- Fulltrain HistGB EXP007 prediction:
  `outputs/predictions/histgb007_fulltrain_test.feather`
- Monthly candidate audit:
  `data/interim/tree_experiments/EXP-BLEND-013-SIX-SOURCE-MONTHS/monthly_comparison.csv`
- Validation/test geometry audit:
  `data/interim/tree_experiments/EXP-BLEND-012-SIX-SOURCE-TRANSFER/result.json`

## Kaggle status

- B001 75/25 TabM+XGBoost: public score 0.131 (current best).
- TabM007 277-feature gain-pruned model: public score 0.130.
- Double-trim B005 is prepared but not uploaded.
- One daily submission remains at the time of this record.

