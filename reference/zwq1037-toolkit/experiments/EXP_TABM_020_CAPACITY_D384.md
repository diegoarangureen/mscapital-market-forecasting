# EXP-TABM-020: relative319 TabM width 384

Date: 2026-09-12

Only `d_block` changed from 256 to 384. The promoted 319 features, seed 42,
Quantile preprocessing, 15 epochs, smooth cosine learning-rate schedule, member
count, block count, dropout, and tree blend were held fixed.

## Deltas versus relative319 width 256

| Window | TabM overall | B001 overall | TabM monthly std | B001 monthly std |
|---|---:|---:|---:|---:|
| 40--49 | -0.001245 | -0.000470 | +0.000314 | -0.000016 |
| 50--59 | -0.000658 | +0.000210 | +0.001021 | +0.000855 |
| 60--70 | -0.002786 | -0.001047 | +0.002344 | +0.001882 |

On 60--70, the wider model also reduced TabM no66 by 0.004261 and 67--70 by
0.004831. The B001 reductions were 0.002469 and 0.002941 respectively. Its worst
month deteriorated sharply.

## Decision

Reject width 384. Keep `d_block=256`. The added features did not create a need
for more capacity; the larger model learned less stable time-specific structure.

