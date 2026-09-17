
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

## 2026-09-16 20:44 CEST — 5-seed ensemble verdict (Track A)
- Seeds 2026/7/42/123/31337, best-epoch val(61-70): 0.14118 / 0.14264 / 0.14415 / 0.14215 / 0.14146. Mean 0.14232.
- Ensemble (mean of best-epoch preds): 0.14287. Below v10 bar (>=0.144 clear) and below best single seed (0.14415).
- DECISION: no submission. Seed ensembling adds ~+0.0005 over mean single, noise-level. Track A (squeeze RealMLP) looking tapped out near val ~0.144.
- Kernel naming lesson: pushing to an existing kernel slug whose dataset list CHANGED fails to mount the new dataset (FileNotFoundError at /kaggle/input/datasets/...). Fresh kernel slug mounts fine. mscapital-seq2-gru-val lineage abandoned; GRU v2 running as mscapital-gru-seq2-val v1.
- GPU spend today: ~1.7h seeds (5x16ens) + GRU2 running. Cumulative ~6.5-7h of 30h week.

## 2026-09-16 21:13 CEST — GRU v2 on seq2: NO SIGNAL (Track B, sequence arm)
- mscapital-gru-seq2-val v1 COMPLETE in ~30min: best val_cos 0.01968 (16 epochs, hid 96, BS 1024, OneCycle 2e-3).
- Same val split (61-70) where RealMLP/246f gets 0.1412. GRU v1 (60x6 every-3rd-bar) was 0.0162; v2 (120x8 per-second) 0.0197.
- VERDICT: raw per-second bar sequences carry ~nothing for the target under direct supervised GRU. Sequence arm CLOSED (also deprioritizes CNN-Transformer on same input).
- Track B remaining: TabM-mini on 246f (diversity), NN-oriented features, research restart for structural ideas.

## 2026-09-16 22:32 CEST — X21 built + 276f val running
- X21 (30 cols, early windows [0,10),[10,30),[30,60): microprice dev, spread, L1 imb, signed vol/amt/frac, order new-pressure, cancel imbalance, net placement). Train 60s+2570s+2609s passes; test fast. All finite, all 30 cols nonzero.
- Dataset diegoaranguren/mscapital-x21 READY (4 files). Val kernel: mscapital-x21d-val v1 (fresh slug), champion config n_ens=16 ep=9, 276f = 246f + X21 hstack (realmlp_val_x21.py).
- Baseline to beat: val 0.141185 / late 0.143986 (same recipe, 246f). Gate: net positive on val AND forward-sims.
- Kernel debug saga: NameError XTRA (defined after use) caused 3 fast ERRORs; fresh-slug mount rule reconfirmed; quota fine (17134s/108000s used, refresh 2026-09-19).
- LB 22:20: unchanged (221 teams, top1 0.172, gate10 0.160, us 173 @ 0.124).

## 2026-09-16 23:29 CEST — X21 (276f) val: POSITIVE, shift sim running
- mscapital-x21d-val COMPLETE: val(61-70) 0.143830 vs baseline 0.141185 => +0.00264. late(66-70) 0.146919 vs 0.143986 => +0.00293.
- Early-window microstructure (10/30/60s) carries real new signal. X22 (drafted) extends the family if shift confirms.
- Shift sim kernel mscapital-x21-shift-val running (tr<=50 -> 51-60; baseline RealMLP 246f shift = 0.133253).
- Gate for refit+submit: shift net positive AND refit val >= 0.144 clear. Submissions 1/5 used, reset ~02:00.

## 2026-09-17 00:14 CEST — X21 CONFIRMED (shift positive)
- Shift sim (tr<=50 -> 51-60): 276f 0.134280 vs 246f 0.133253 => +0.00103. All gates positive: val +0.00264, late +0.00293, shift +0.00103.
- X21 included. Matrix champion now 276f. Next: X22 (early-window dynamics + cross-stream) to chain gains before refit+submission; 276f val 0.14383 alone is under the 0.144-clear submission bar.

## 2026-09-17 00:59 CEST — COS-DOMINANT LOSS: big gain
- lambda_cos 0.01 -> 1.0 on 276f: val 0.147721 (+0.00389 vs 0.143830), late 0.153287 (+0.00637 vs 0.146919). Clears the 0.144 bar on val.
- Source of idea: top-team levers (direct cosine optimization). Shift sim running (mscapital-cos-shift-val). If positive -> strong v10 candidate (refit 0-70, 276f, cos loss).
- x22-303f first run ERRORED (mount race suspected); re-pushed wrapped as mscapital-x22w-303f-val.
- LB 00:59: unchanged (221, top1 0.172, gate10 0.160, us 173 @ 0.124).

## 2026-09-17 01:46 CEST — cos loss CONFIRMED, v10 refit launched
- Shift sim cos+276f: 0.136149 vs 0.133253 (246f mse) / 0.134280 (276f mse) => +0.0019. All gates positive.
- Cos loss evidence: val 0.147721 / late 0.153287 / shift 0.136149. Gate (val>=0.144 + robustness) PASSED.
- v10 = refit 0-70, 276f, lambda_cos=1.0, n_ens=16, ep=9: kernel mscapital-v10-refit-x21cos RUNNING (ETA ~02:25). On COMPLETE: download submission.csv + test_pred.npy, verify (647896 rows, no-data zeroed, clipped), submit via blob flow.
- x22 NameError XTRA2 fixed (definition order); 303f val re-running as mscapital-x22f-303f-val (old loss) to measure X22 marginal value.
- LB 00:59 unchanged (221, top1 0.172, gate10 0.160, us 173 @ 0.124).

## 2026-09-17 02:23 CEST — X22 positive (old loss); v10 refit re-running; 303f+cos queued
- X22 (303f, lambda_cos 0.01): val 0.145410 (+0.00158 vs 276f), late 0.148259 (+0.00134). Positive; shift sim pending.
- v10 refit first run ERRORED fast, cause unknown (transient suspected - identical wrapped script now RUNNING past the failure point as mscapital-v10w-refit).
- Launched mscapital-x22cos-303f-val: 303f + cos loss. If > 0.14772 clear, v11 = 303f+cos refit.

## 2026-09-17 02:56 CEST — 303f+cos best yet; v10b refit + shift running
- x22cos (303f + cos): val 0.149518 (+0.00180 vs cos276f), late 0.154895 (+0.00161). X22 adds on top of cos loss.
- v10 refit failure cause: no competition_data_sources attached -> submission.csv template glob empty (IndexError line 244). Fixed: push with comps=['ms-capital-real-financial-market-forecasting']; push_kernel now takes comps=.
- RUNNING: mscapital-v10b-refit (276f+cos refit 0-70, with comp source) + mscapital-x22cos-shift (303f+cos, tr<=50->51-60).
- Plan: v10b completes ~03:35 -> submit (gated: all 3 positive). If x22cos shift positive -> v11 = 303f+cos refit (realmlp_test_x22cos.py) -> 2nd submission.

## 2026-09-17 03:41 CEST — v10 SUBMITTED (ref 56291244), v11 refit running
- v10 = 276f+cos refit 0-70. Submission verified: 647896 rows, 17 no-data zeros, clipped, no NaN. Blob flow OK (INBOX blob type).
- x22cos shift: 0.137157 vs cos276f shift 0.136149 => +0.0010. X22 CONFIRMED all gates. 303f+cos: val 0.149518/late 0.154895/shift 0.137157.
- v11 = 303f+cos refit 0-70: kernel mscapital-v11-refit-x22cos RUNNING (ETA ~04:20). Submit after v10 LB lands (transfer-gap read).

## 2026-09-17 04:26 CEST — v10 LB FLAT: 0.124 (ref 56291244)
- v10 (276f+cos, val 0.1477/+0.0065 vs v9) scored public LB 0.124 — identical to v9 at 3-decimal precision.
- READING: transfer gap ate the whole val gain (v9 gap 0.0172; v10 gap ~0.024). Val 61-70 gains do NOT linearly transfer to test. The bottleneck is now shift-robustness, not val.
- v11 (303f+cos refit) artifact READY but HELD: its val edge over v10 (+0.0018) is below LB resolution given v10's flat transfer; submitting it would be LB-probing, against the submissions rule.
- New priority: ideas that target the transfer gap directly (latest-window validation for selection, shift-robust feature choice, regime-conditional models).

## 2026-09-17 04:28 CEST — adversarial validation + advdrop experiment
- Adversarial AUC late-train(66-70) vs test on X21+X22 (57f): 0.6954. Real distribution shift.
- Top discriminators: spread family (x22_d_spr_0L, x21_m2_spread, x21_m1_spread), imb, book slopes, tx vol. => spread/vol regime differs between late-train and test. Explains flat v10 transfer.
- Per-month AUC vs test: 0.63-0.82, no monotone time trend (month 50 most different 0.82; 53/56/68 least). Test is not simply "a later month".
- Experiment launched: mscapital-advdrop-val = 303f+cos minus top-5 adversarial discriminators (idx 301,267,268,299,257). If val holds >=~0.1485, those cols were regime noise and the model should transfer better -> refit+submit candidate.

## 2026-09-17 05:12 CEST — advdrop: cheap robustness
- Drop top-5 adversarial cols: val 0.148940 (-0.00058), late 0.154390 (-0.00051) vs full 303f+cos. Adversarial AUC 0.6954 -> 0.6568.
- Shift sim running (mscapital-advdrop-shift). Gate: >= ~0.136 (cos276f shift 0.136149). If pass -> refit advdrop 303f (realmlp_test_x22cos.py + DROP) -> submit as v11.

## 2026-09-17 05:56 CEST — advdrop PASS, v11 refit running
- advdrop shift 0.137205 (best shift reading; cos276f 0.136149, x22cos 0.137157). Gates: val 0.148940 / late 0.154390 / shift 0.137205 / advAUC 0.657.
- v11 = advdrop 303f+cos refit 0-70: kernel mscapital-v11b-refit-advdrop RUNNING (ETA ~06:35). Then verify + submit (2/5).

## 2026-09-17 06:22 CEST — v11 SUBMITTED (ref 56294572)
- v11 = advdrop 303f+cos refit 0-70. Verified: 647896 rows, 17 no-data zeros, clipped [-0.222, 0.230], no NaN.
- Awaiting score. If flat again at 0.124: transfer gap robust to feature-regime dropping too -> next lever is model-side (holdout+early-stop submission instead of refit, per bestwater intel) or regime-conditional.

## 2026-09-17 07:03 CEST — v12 SUBMITTED: holdout+ES (ref 56295402)
- Kernel mscapital-holdout-advdrop COMPLETE: RealMLP 303f-5adv cos-loss, train<=60 with early stop on 61-70, predicts test with best-epoch EMA (best_val_cos 0.14894, matches advdrop val run).
- Submission verified: 647896 rows, no NaN, 17 zeroed no-data rows, clipped, corr 0.919 vs v11 refit preds.
- Parent authorized 06:25 ("Luz verde para la submission holdout cuando el kernel termine").
- HYPOTHESIS (bestwater intel): holdout+early-stop beats refit-on-full in this comp. v9/v10/v11 refits all flat at 0.123-0.124. If v12 > 0.124, the whole refit procedure changes.
- seq3 (120x12 engineered per-second channels incl. order-flow stream) building locally; GRU val next.

## 2026-09-17 07:18 CEST — v12 SCORED: 0.124 (ref 56295402) — holdout+ES FLAT
- v12 (holdout<=60 + early stop, best-epoch EMA, no refit) scored public LB 0.124.
- VERDICT: holdout+ES == refit-on-full within LB resolution. Full family: v9 0.124, v10 0.124, v11 0.123, v12 0.124.
- bestwater intel ("holdout+ES beats refit here") did NOT produce an LB jump. The procedure is NOT the bottleneck.
- CONCLUSION: the 0.123-0.124 plateau is structural for this feature family/model class. Val gains 0.141->0.149 never transferred. Next levers: different signal source (seq3 engineered per-second GRU, building), jump-research track (papers/repos), shift-targeted validation for model selection.
- Submissions used today: 1 (v12). Remaining: 4.

## 2026-09-17 08:13 CEST — seq3 datasets LIVE + GRU val kernel pushed
- Datasets: mscapital-seq3-train + mscapital-seq3-test (120 steps x 12 channels, uniform 1s grid, market+tx+ORDER streams; first time the order-flow stream enters a sequence model here).
- Channel 5 (tx signed volume) stored as signed log1p after float16 overflow fix.
- Kernel diegoaranguren/mscapital-seq3-gru-val pushed (1-layer GRU + MLP head per JS8 lesson). Val protocol tr<=60 -> 61-70. Verdict expected ~09:00.
- If seq3 val ~0.02 like seq1/seq2: sequence arm is definitively dead with three independent builds; jump research pivots fully to tabular shift-robustness + community mining via cloud browser.

## 2026-09-17 08:58 CEST — seq3 VERDICT: val_cos 0.0364. SEQUENCE ARM DEAD.
- seq3 GRU (120x12 engineered per-second channels, market+tx+order streams): best val_cos 0.0364.
- Lineage: seq1 raw GRU 0.016, seq2 120x8 market-only engineered 0.0197, seq3 +order-flow channels 0.0364.
- Order-flow stream nearly doubled sequence signal (0.0197 -> 0.0364) but tabular RealMLP sits at 0.149 val. Sequences capture ~25% of tabular signal at best; ensemble contribution negligible.
- DECISION: sequence arm closed after three independent builds. Discard criterion documented: val_cos < 0.10 (would need >= ~0.10 to justify ensemble weight).
- Jump track next: (a) harder adversarial feature pruning (iterative drop by adv-AUC contribution, measure shift-sim), (b) MSCapital discussion mining via cloud browser (bestwater intel source), (c) UnseenAnchor ensemble intel re-read.

## 2026-09-17 11:11 CEST — SANDBOX WIPE + RECOVERY
- Sandbox rebuilt between 09:00 and 09:42 (all /tmp state lost). Detected 09:42, recovered via recover.sh (repo clone, pip, 8.6GB feathers re-downloaded). RESTORE_DONE 11:10 (matrices + X21/X22 npys re-downloaded from Kaggle datasets).
- Hardening: streamcol2.py and adv_iter.py were local-only at wipe; both now committed to repo (src/). All datasets/kernels unaffected (Kaggle-side).
- adv_iter (iterative adversarial pruning, 303f, late-train 66-70 vs test, k=5/10/15/20/30) relaunched 11:11.

## 2026-09-17 13:31 CEST — adv_iter RESULT: pruning exhausted (negative)
- Adversarial AUC late-train(66-70) vs test, 303f: full 0.7981; drop10 0.7905; drop20 0.7849.
- Top adversarial features: X2 vol/spread level stats (mk_mid_vol, tx_px_std, relspread_mean, px_vol, vol_total, spread_mean), X13 group-distance, X16 cancel-distance, x22 slope dynamics.
- READING: shift signal is DIFFUSE - removing 20 features barely reduces separability. The whole feature distribution moved (regime level), not a few bad columns. Feature pruning cannot fix transfer. Consistent with v11/v12 flatness.
- Discard criterion: adversarial pruning as transfer fix is closed; advdrop-5 stays in the champion only because it didn't hurt val.
- Next structural candidate: DANN-style domain-adversarial head on RealMLP (gradient reversal vs month/regime classifier) - forces regime-invariant representation; measurable in val + shift-sim. NOT on UnseenAnchor negatives list (CORAL/SSL is, DANN isn't).

## 2026-09-17 16:15 - mhmlp-v2 (multi-head MLP 455f+XS75) result
- Val cos full(61-70): 0.159045, late(66-70): 0.165921. Beats old champion (0.148940/0.154390) but LOSES to x455 RealMLP (0.167133/0.174921).
- v1 died with blank-log ERROR (likely OOM); v2 (K_HEADS 32, BS 1024) ran clean.
- Ensemble test: corr(x455, mhmlp)=0.848; weighted ensembles (w 0.5-0.7) all WORSE than pure x455 (0.1592-0.1595 vs 0.1671). mhmlp too weak to add diversity value.
- Verdict: multi-head MLP is not the jump. RealMLP (PBLD+NTP) remains best arch on 455f. kfold455 (15-model fold-avg) is the v13 candidate, running.

## 2026-09-17 19:30 - XS75 built + 0726 alignment verified
- build-xs75b (CPU kernel): XS75 = 30 within-month pct-rank + 15 within-month z + 30 cross-feature rank, LGBM-gain top30 selection from 298f-advdrop base. Train within-month; test GLOBAL (no month labels exist for test - matches bestwater's global-test variant).
- CRITICAL verification: yunsuxiaozi rfmf-0726data sample_ids are SHUFFLED but are an exact permutation of 0..n-1 for both train (1,257,637) and test (647,896). sort_values('sample_id') restores alignment - validates x455 val result AND kfold455/v13 test features.
- Own private kernel output mounts work as kernel_data_sources.
- mhmlp-v1 root cause: dataset path mismatch (lineage-pinned mounts). Kernel logs arrive as JSON string - json.loads before iterating.
- Next: mscapital-x530-val running (champion + 152 public + XS75 = 530f, val tr<=60/61-70).

## 2026-09-17 21:00 - x530 (champion + 152 public + XS75): val 0.173447 / late 0.181450
- XS75 adds +0.0063 val over x455 (0.167133/0.174921). Second consecutive val breakthrough layer.
- Ladder now: 298f 0.1489 -> +152 public 0.1671 -> +XS75 0.1734. Matches bestwater's 689f construction (462 own + 152 public + 75 XS) minus his extra own features.
- corr(x530, x455) = 0.958; ensembles of the two are worse than pure x530 (x530 dominates).
- Implication: v14 candidate = kfold protocol on 530f (once v13 LB lands and a GPU slot frees).

## 2026-09-17 22:25 - v13 LB 0.139 (rank 118/223) - TRANSFER CONFIRMED
- kfold455 (RealMLP 455f, 5 purged folds x 3 seeds, holdout+ES, 15-model avg): LB 0.139, +0.015 over the 0.123-0.124 plateau. Rank 173 -> 118. Gate10 0.160, top1 0.172.
- Kernel OOF: 61-70 0.1678, 66-70 0.1745. Per-fold val cos 0.1419-0.1724 (later folds stronger). 15/15 models trained, 24465s total.
- First val->LB transfer of the project. The 0726 public features broke the ceiling exactly as external evidence (bestwater 0.142) suggested.
- Alignment of 0726 features verified by permutation assert before submitting.
- Submissions today: 4/5 used (v10, v11, v12, v13). 1 left, reserved.
- v14 candidate launched 22:29: mscapital-kfold530 (15-model fold-avg on 530f = +XS75; x530 val was 0.173447/0.181450).

## 2026-09-18 00:02 - X23 built (148 stream aggregates), x678 val launched
- X23: 12 stats x 12 channels (mean/std/min/max/q10/q90/last30/first30/diff/std_last30/ac1/ac5) + 4 cross-channel correlations from the seq3 order-flow grid. Built from existing seq3 datasets (CPU, 4 min). Files in kernel output mscapital-build-x23.
- mscapital-x678-val running: champion recipe on 530f+X23 = 678f, val tr<=60/61-70. Baseline to beat: x530 0.173447/0.181450.
