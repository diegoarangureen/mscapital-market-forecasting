# EXP-TREE-069: relative319 in XGBoost

Date: 2026-09-12

The same twelve relative-scale features were added to the fixed EXP053R XGBoost
(800 trees, depth 5, column sampling 0.8, CUDA). Three forward windows were run.

## Pure XGBoost deltas

| Window | Overall | Monthly std | Worst | q25 | no66 | 67--70 |
|---|---:|---:|---:|---:|---:|---:|
| 40--49 | +0.000541 | -0.001208 | +0.002747 | +0.000330 | -- | -- |
| 50--59 | +0.000927 | -0.001621 | -0.000567 | +0.002980 | -- | -- |
| 60--70 | +0.001802 | +0.001783 | +0.002116 | +0.000200 | +0.000139 | +0.000849 |

## Blend deltas when only the tree component changes

The TabM component is held at relative319 and the tree remains at 25% weight.

| Window | Overall | Monthly std | Worst | q25 | no66 | 67--70 |
|---|---:|---:|---:|---:|---:|---:|
| 40--49 | +0.000005 | -0.000092 | +0.000567 | -0.000583 | -- | -- |
| 50--59 | +0.000065 | -0.000445 | -0.000195 | +0.002102 | -- | -- |
| 60--70 | +0.000279 | +0.000405 | +0.000368 | -0.000050 | -0.000102 | +0.000021 |

## Decision

The features help standalone XGBoost, but almost all of the gain is redundant
with the stronger relative319 TabM. The blended 60--70 no66 score falls slightly
and late monthly variation rises. Keep the original EXP053R tree in the preferred
75/25 candidate; do not full-train relative319 XGBoost yet.

