# X30 post-screen contingency plan (written 2026-09-24 16:50, pre-verdict)

Baselines (kfold TPU seed2026 per-fold, screen reference):
f1 0.143251, f2 0.149152, f3 0.148980, f4 0.155356, f5 0.171549.
Noise: XLA session ~0.003, val fold-std ~0.004. Verdict rule: consistent shift >0.002 across folds = signal; else FLAT.

## If SIGNAL (X30 panel beats baselines consistently)
1. Ablate inside X30 (36f -> groups) to find the load-bearing family before spending subs:
   OFI buckets (6), trade-sign autocorr (3), RV/Parkinson/semivol/vol-of-vol (8), order arrival/cancel (4), QI dynamics (6), microprice/mid slopes (4), spread/book shape (3), trade ticker vwap_dev/tvol/tcount (3).
   Each ablation = one 5-fold panel (~1.5-2h TPU). Order: drop-one-group on the 3 most promising groups first.
2. If a single family carries >50% of the gain, extend THAT family (more lags/buckets), re-screen.
3. Only then consider a submission candidate: full champion pipeline + winning X30 subset, forward-sim gated. Submission requires parent authorization (val >= 0.130 claro + robustez verificada).

## If FLAT (within noise, no consistent shift)
1. Confirm not an integration bug: check per-fold val curves, feature importance dump, sanity of X30 columns (names order matches build).
2. Diagnosis branches:
   a. Signal absent at 60s book/transaction streams -> microstructure at this horizon carries nothing beyond price/volume aggregates. Pivot research to the 600s embedded trade ticker (market.feather) aggregates and cross-sample structure (NOT cross-sectional XS - thrice-confirmed LB trap; absolute per-sample only).
   b. Signal present but RealMLP can't use it -> try the same 36f with the GRU sequence head (best public single 0.143 was GRU) on the raw streams instead of aggregates.
3. No submissions either way without parent authorization.

## If FAILED (infra)
- Read log, fix, re-push as v3. Known failure modes: dataset slug-prefix mounts (fixed v2), TPU OOM on 486f (batch 256 fits X29), async mount layout (handled).
