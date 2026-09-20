# EXP-TABM-016--018: relative-scale feature checks

Date: 2026-09-12

## Fixed protocol

- Baseline: 307-feature Quantile TabM, smooth cosine learning rate, epoch 15.
- Same architecture, target scaling, seed 42, batch size, and XGBoost 75/25 blend.
- Forward windows: 0--39 -> 40--49, 0--49 -> 50--59, 0--59 -> 60--70.
- The twelve candidates are target-free and computed independently inside each sample.

## Feature groups

- `price8`: last mid relative to full/recent mean, four relative-spread summaries,
  and mean/last transaction price relative to mid.
- `activity4`: recent/full transaction count and volume shares, plus full-window
  peak count and peak volume shares.
- `add12`: the union of `price8` and `activity4`.

## Mean TabM deltas versus the 307-feature baseline

| Candidate | 40--49 overall | 50--59 overall | 60--70 overall | 60--70 no66 | 67--70 |
|---|---:|---:|---:|---:|---:|
| add12 | +0.001572 | +0.000139 | -0.000446 | +0.000775 | +0.000364 |
| price8 | +0.001637 | -0.000344 | -0.000031 | -0.000222 | -0.000017 |
| activity4 | -0.000084 | -0.000503 | -0.000892 | -0.000858 | -0.001317 |

## Mean B001 deltas versus the 307-feature baseline

| Candidate | 40--49 overall | 50--59 overall | 60--70 overall | 60--70 no66 | 67--70 |
|---|---:|---:|---:|---:|---:|
| add12 | +0.001256 | +0.000106 | -0.000508 | +0.000487 | +0.000169 |
| price8 | +0.001184 | -0.000333 | -0.000102 | -0.000210 | -0.000055 |
| activity4 | -0.000083 | -0.000462 | -0.000865 | -0.000743 | -0.001130 |

## Stability observations

- `add12` reduced TabM monthly standard deviation in all three windows by
  0.000223, 0.001558, and 0.000518 respectively.
- `add12` improved the worst month in all three windows by 0.002249, 0.005425,
  and 0.002947 respectively.
- `price8` and `activity4` do not reproduce the union's robust late-window gain.
  The groups therefore cannot be selected independently from seed-42 results.

## Decision

`add12` is a provisional candidate, not yet a promoted feature set. It improves
four of the five main forward/generalization scores but loses 0.000446 on the
60--70 overall score. Because changing the input dimension also changes TabM's
initial parameter draw, a paired second-seed confirmation is required before
full-data training or submission.

