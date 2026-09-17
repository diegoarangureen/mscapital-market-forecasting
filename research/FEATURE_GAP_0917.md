# Feature gap analysis (post-v13, Sept 17 22:30)

## Where we stand
- Ladder: 298f 0.124 LB (plateau) -> 455f (+152 public 0726) LB 0.139 -> 530f (+XS75) val 0.1734, v14 pending.
- bestwater full recipe: 462 OWN + 152 public + 75 XS = 689f, TabM, LB 0.142.

## The remaining structural gap: his 462 own features are PRIVATE
- His kernels (kgpu-tabm-cos689-3seed, ktpu-tabm-cos-v6-3seed) both load bestwater/mscapital-lgb-features (X_train_features.npy 462f). Dataset GET -> 403 (private).
- His public kernel list has NO feature-generation kernel for MSCapital. Cannot inspect or reimplement his 462 directly.
- Conclusion: closing the remaining gap requires OUR OWN feature engineering expansion, not mining.

## TabM arch: deprioritized
- mhmlp (our TabM-like multi-head MLP, 32 heads) on 455f+XS75: val 0.1590 vs RealMLP 0.1671 same features. Ensemble degrades. RealMLP (PBLD+NTP) stays champion arch. Do not port TabM.

## X23 candidate (next feature family): order-flow stream tabular aggregates
- seq3 evidence: adding the order-flow stream DOUBLED sequence signal (0.0197 -> 0.0364). That stream never entered the tabular features.
- src/streamcol2.py already computes stream columns; aggregate into per-sample tabular stats (mean/std/skew/quantiles of OFI, cancel ratios, depth slopes at multiple windows) = X23.
- Priority after v14 result.

## Compliance notes
- Using PUBLIC comp-shared feature artifacts (yunsuxiaozi 0726) validated by precedent; bestwater's 462 are private = off-limits anyway.
- Never blend others' prediction files (yangq369 trap).
