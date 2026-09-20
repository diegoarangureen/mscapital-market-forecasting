# EXP-TABM-015 Quantile TabM Kaggle result

Date: 2026-09-12

- Kaggle submission ref: `56177561`
- File: `tabm_quantile_coslr15_tree053r_blend75_fulltrain_trim1.csv`
- Description: `Quantile TabM trim1 cosine-LR15 + 25% XGB053R`
- Status: complete
- Public score: **0.134**
- Previous best B001: 0.131
- Absolute displayed improvement: **+0.003**
- Maximin: 0.130

The result supports the multi-window local evidence that training-fold empirical
normal-quantile preprocessing transfers better than B001's mean/std scaling.
The exact improvement is not known beyond the three decimals displayed by the
Kaggle CLI.

Local forward comparisons for the frozen smooth-LR epoch-15 policy:

| Forward window | Fixed-LR Quantile B001 | Smooth-LR Quantile B001 | Change |
|---|---:|---:|---:|
| 0-39 -> 40-49 | 0.149006 | 0.149831 | +0.000824 |
| 0-49 -> 50-59 | 0.153265 | 0.154156 | +0.000890 |
| 0-59 -> 60-70 | 0.177335 | 0.177125 | -0.000210 |

The submitted test prediction had cosine agreement 0.956955 with B001, so it
was a material but not unrelated change. Three daily submissions remained
immediately after upload.
