# Kaggle Transformer sources

This folder contains the exact Kaggle-oriented sources for the strongest standalone Transformer line and the later event-residual variant.

## Factorized Transformer temporal 3-fold

- Public leaderboard score: **0.145**.
- Temporal cutoffs: months 52, 57, and 62.
- One seed per cutoff and four epochs per model.
- Trained with two Tesla T4 GPUs.
- Final prediction: equal average of the three temporal models.

## Market-conditioned event-residual Transformer

This later variant was used in the final 0.152 blend. The final Transformer slot used 40% raw Factorized Transformer and 60% event-residual Transformer.

## Inputs

The included cache builder creates sequence-grid caches from competition files. The model also expects Relative319 and order/quote-position static features; their feature-generation code is in the main scripts archive. Competition data, derived matrices, checkpoints, and predictions are not redistributed. Update original /kaggle/input paths when using different Dataset or Notebook slugs.
