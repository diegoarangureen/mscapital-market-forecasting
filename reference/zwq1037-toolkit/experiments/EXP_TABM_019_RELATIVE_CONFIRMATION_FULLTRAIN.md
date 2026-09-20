# EXP-TABM-019 and full-data relative319 candidate

Date: 2026-09-12

## Paired seed-137 confirmation

The 307-feature baseline and 319-feature add12 candidate were retrained with the
same seed 137, Quantile preprocessing, epoch-15 budget, and cosine learning-rate
schedule. This controls for the change in the first-layer parameter draw caused
by changing the input dimension.

| Window and metric | TabM mean delta | B001 mean delta |
|---|---:|---:|
| 50--59 overall | +0.001344 | +0.001256 |
| 60--70 overall | +0.000876 | +0.000807 |
| 60--70 no66 | +0.000955 | +0.000738 |
| 67--70 | +0.000056 | +0.000107 |
| 60--70 monthly std | -0.000155 | -0.000042 |
| 60--70 monthly q25 | +0.000711 | +0.000599 |

The second seed confirms the add12 direction. Together with seed 42, the feature
set is promoted for the TabM component.

## Full-data candidate

- Training months: 0--70.
- Features: 319.
- Epochs: 15 with the frozen smooth cosine schedule.
- Training time: 187.475 seconds on CUDA.
- Prepared mean and trim1 submissions; neither was uploaded automatically.
- Preferred file: `outputs/submissions/tabm_quantile_rel319_coslr15_tree053r_blend75_fulltrain_trim1.csv`.
- The new trim1 prediction has cosine agreement 0.989922 with the submitted
  307-feature Quantile trim1 candidate and relative L2 difference 0.141909.

