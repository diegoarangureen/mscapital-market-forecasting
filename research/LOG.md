
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

## RealMLP shift forward-sim (2026-09-16 13:20) — POSITIVE
tr<=50 -> val 51-60. best val_cos 0.133253 (ep9) vs GBM shift ref 0.128442: +0.0048 CLEAR.
Curve healthy (rises to ep9-10). late_cos n/a (no months>=66 in this split).
V9 DECIDED: RealMLP solo refit 0-70, 9 epochs (val curve best), submission pending refit kernel.

## V9 SUBMITTED (2026-09-16 15:00) — LB 0.124, rank 173/220
RealMLP solo refit 0-70, 9 epochs, n_ens=16, submission ref 56279103.
LB 0.116 -> 0.124 (+0.008): val gain (+0.009) transferred almost 1:1. Champion v7/v8 replaced.
17 no-data rows zeroed, clip q0.1/99.9 of train preds. LB: 220 teams, top1 0.172, top10 gate 0.159.
Gap to gate: 0.035. Next: seed-ensemble RealMLP (GPU cheap), n_ens/epochs ablation, TabM, features for NN.

## Seed ensemble val (2026-09-16 16:20) — MARGINAL +
3 seeds (2026/7/42), RealMLP val 61-70. Singles (last epoch): 0.14041/0.14264/0.14311; best-epoch bests: 0.14118/0.14264/0.14415. Ensemble of unit-normalized last-epoch preds: 0.14317.
Ensemble vs single-model best ~ +0.002. Positive direction, weak magnitude. Plan: 5-seed ensemble of BEST-epoch preds as v10 candidate (expect val ~0.144-0.145).

## GRU seq v1 (2026-09-16 16:25) — NO SIGNAL, build too thin
CNN+GRU over 60x6 every-3rd-bar market sequences: val_cos 0.0162 after 12 epochs (rising but tiny). Hypothesis: 6-channel market-only subsampled sequences lack the information UnseenAnchor's seq models used (per-second resampled ALL streams). Also check zero-sequence coverage. Not abandoned, needs richer build; deprioritized vs RealMLP squeeze.

## Sandbox wipe #4 (2026-09-16 ~16:24)
All local state lost again; zero work lost (Kaggle dataset + GitHub). Kernels unaffected. Recovery rerun from repo.

## 2026-09-16 20:27 CEST — seq2 dataset + seq-gru v2 kernel
- Dataset diegoaranguren/mscapital-seq2 READY: seq2_train.f16 (2414663040B), seq2_test.f16 (1243960320B), 1257637/647896 samples x 120 steps x 8ch f16.
- seq_gru_val.py patched for seq2 (STEPS=120, CH=8, EPOCHS=16, TAG=seq2); kernel pushed as mscapital-seq2-gru-val v3, GPU, datasets: mscapital-matrices + mscapital-seq2.
- GRU v1 (60x6, every-3rd-bar market-only) had NO signal (val 0.016). v2 test: per-second 120x8 sequences.
- kaggle_io.py: added push_kernel/kernel_status/download_kernel_file/leaderboard helpers.
- LB 19:46: 221 teams, top1 0.172, gate10 0.160, us 173 @ 0.124 (unchanged).
