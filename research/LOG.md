
## X20 verdict (2026-09-16 ~11:25) — DISCARDED
Window-deltas + cross-stream interactions from X2 stats (UnseenAnchor-inspired, cheap subset).
- x2x20 (X2+X20): 0.125684 vs X2 0.123587 → +0.0021 (solo axis positive)
- xall20 (246+X20) full val 61-70: 0.131353 vs 0.132116 → -0.0008 NEGATIVE
- xall20 shift (tr<=50 -> 51-60): 0.129434 vs 0.128442 → +0.0010 (within LB resolution 0.001)
Criterion: include only if full val AND forward-sims net positive. Full val fails -> X20 discarded.
Learning: deltas/interactions help in a small feature set but dilute in the full 246 matrix; shift gain is noise-level.

## RealMLP GPU val (2026-09-16 12:36) — STRUCTURAL JUMP
Kernel diegoaranguren/mscapital-realmlp-val (P100, n_ens=16, 10 ep, ~20 min total).
Epoch curve val_cos: ep1 0.0960 -> ep5 0.1312 -> ep9 0.1412 (best) -> ep10 0.1404.
best_val_cos 0.141185 | late_cos (66-70) 0.143986 @ ep9.
vs GBM 246f val ref 0.132116: +0.009 CLEAR (>> LB resolution 0.001). First architecture to beat GBM on full val.
Pending: shift forward-sim (tr<=50 -> 51-60, GBM ref 0.128442), blend val with GBM+MLP.

## Blend analysis GBM x RealMLP (2026-09-16 12:46)
Val 61-70, unit-normalized preds. GBM 246f (2-seed regen): 0.13262 val / 0.13320 late. RealMLP: 0.14119 / 0.14399.
Corr(RealMLP, GBM) = 0.9028 (high). Blend grid w_rlm 0.3..0.7: best 0.14153 @ 0.7 — +0.0003 vs solo, noise-level.
Conclusion: RealMLP dominates; blend adds nothing meaningful. v9 candidate = RealMLP solo refit 0-70 (9 epochs, best from val curve) pending shift check.
