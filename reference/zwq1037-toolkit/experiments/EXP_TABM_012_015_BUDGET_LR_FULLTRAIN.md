# EXP-TABM-012 to 015: third window, budget, LR, and fulltrain

Date: 2026-09-12

## Matched-budget quantile result

The earlier nested gate uses 0-29 / 30-39 for epoch selection and 0-39 / 40-49
for formal evaluation. Because the selected epochs differed greatly, the same
preprocessors were also compared at matched 6 and 16 epoch budgets.

| Epochs | Standard TabM | Quantile TabM | Change | Standard B001 | Quantile B001 | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.132356 | 0.133508 | +0.001152 | 0.137185 | 0.138278 | +0.001093 |
| 16 | 0.146049 | 0.149455 | +0.003406 | 0.147185 | 0.149644 | +0.002459 |

Most of the initially observed +0.017 jump was caused by the epoch budget, but
the quantile preprocessing itself remains positive at both matched budgets.

## Fixed learning-rate curves

With a constant 0.002 learning rate, curves are non-monotonic and often show a
second improvement after four or more stale epochs. Patience=4 can stop too
early. Best fixed-LR B001 epochs are 20, 11, and 15 on the three forward
windows. Neither a single epoch nor approximate matched gradient steps is
uniformly optimal.

## Smooth cosine learning rate

The single tested schedule decays smoothly from 0.002 to 0.0001 over 20 epochs.
At the same epoch 15:

| Forward window | Fixed LR B001 | Smooth LR B001 | Change |
|---|---:|---:|---:|
| 0-39 -> 40-49 | 0.149006 | 0.149831 | +0.000824 |
| 0-49 -> 50-59 | 0.153265 | 0.154156 | +0.000890 |
| 0-59 -> 60-70 | 0.177335 | 0.177125 | -0.000210 |

On 60-70, smooth epoch 15 still improves the diagnostic slices versus fixed
epoch 15: no-66 0.153990 -> 0.154756 and 67-70 0.153295 -> 0.153422. It also
prevents the severe fixed-LR collapse at epoch 20. The frozen full-data policy
is therefore the 20-epoch smooth schedule stopped after epoch 15.

## Full-data candidates

The selected policy was trained on all labelled months 0-70. Training took
192.15 seconds on GPU. Test features were loaded only after releasing the full
training feature matrix.

- Mean: `outputs/submissions/tabm_quantile_coslr15_tree053r_blend75_fulltrain_mean.csv`
- Trim1: `outputs/submissions/tabm_quantile_coslr15_tree053r_blend75_fulltrain_trim1.csv`
- Both contain 647,896 aligned IDs, finite predictions, and no duplicates.
- Trim1 has cosine agreement 0.956955 with submitted B001 and relative
  prediction-difference norm 0.2932.
- Status: prepared, not uploaded to Kaggle.

## Artifacts

- `scripts/exp_tabm_012_early_forward_preprocessing.py`
- `scripts/exp_tabm_013_preprocessing_budget_crosscheck.py`
- `scripts/exp_tabm_014_quantile_stage_curves.py`
- `scripts/exp_tabm_015_quantile_cosine_lr_curves.py`
- `scripts/train_full_quantile_tabm_b001_submission.py`
- `data/interim/tree_experiments/EXP-TABM-012-EARLY-FORWARD-PREPROCESSING/`
- `data/interim/tree_experiments/EXP-TABM-013-PREPROCESSING-BUDGET-CROSSCHECK/`
- `data/interim/tree_experiments/EXP-TABM-014-QUANTILE-STAGE-CURVES/`
- `data/interim/tree_experiments/EXP-TABM-015-QUANTILE-COSINE-LR-CURVES/`
- `outputs/submission_metadata/tabm_quantile_coslr15_tree053r_blend75_fulltrain.json`
