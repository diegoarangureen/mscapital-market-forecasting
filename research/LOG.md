
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

## 2026-09-18 01:32 - x678 (530f + X23 stream aggregates): val 0.175217 / late 0.184091
- +0.0018 over x530 (0.173447/0.181450). Diminishing but positive; the order-flow stream helps tabular too (as seq3 hinted).
- Val ladder: 298f 0.1489 -> +152 public 0.1671 -> +XS75 0.1734 -> +X23 0.1752.
- corr(x678,x530)=0.978; ensemble ~flat vs pure x678.
- Question for v14/v15: kfold530 lands ~03:00; kfold678 would be the stronger submission but costs another ~5h GPU. Decision point at kfold530 completion.

## 2026-09-18 06:50 - kfold530 kernel complete (v14 candidate)
- OOF cos: 61-70 0.170137, 66-70 0.178068, 40-64 0.149967. vs kfold455 (v13): +0.0023/+0.0035 - consistent with val ladder.
- 15/15 models, 28571s. Submission.csv verified (647896 rows, ids match, 17 nodata zeros).

## 2026-09-18 06:51 - v14 (kfold530, +XS75): LB 0.135 - NEGATIVE vs v13 0.139
- XS75 gains in val (+0.006) and kfold OOF (+0.0023) but LOSES 0.004 on LB. Cross-sectional features (within-month train ranks, global-test) do not transfer to the public test regime.
- Second family showing val/OOF vs LB divergence; lesson: features with cross-sectional/transductive computation need the same computability story on test, and even then may not transfer.
- Public LB stays 0.139 (Kaggle keeps best submission). v13 remains champion artifact.
- v15 (kfold678c incl XS75) still running for the data point but disfavored for submission under this evidence.
- GPU QUOTA EXHAUSTED (30h weekly, hit at x603-val push 06:52; refresh ~Sep 19). x603 val (455f+X23, no XS) queued. kfold678c already running continues to ~10:50.

## 2026-09-18 09:05 - X24 built (41 absolute per-sample microstructure features)
- CPU kernel mscapital-build-x24b (~7 min/split): log-volume moments, signed-flow moments and drift, spread/imbalance volatility per early/mid/late bucket, order cancel/new dynamics. All absolute per-sample (no cross-sectional/transductive computation) - designed to be LB-safe under the XS75 lesson.
- Dataset diegoaranguren/mscapital-x24 created (train 1257637x41, test 647896x41, float32, no NaNs).
- One broadcasting bug (signed-flow-drift dividing N3 vector by ns slice) fixed after first push errored; logs fetched via list_kernel_session_output.
- Next: val 455f+X23+X24 (and 455+X24) once GPU quota refreshes (~Sep 19). kfold678c still running (analysis only).

## 2026-09-18 10:27 - GPU quota confirmed: refresh 2026-09-19T00:00:00Z (02:00 CEST)
- GetAcceleratorQuotaStatistics: GPU used 126936s / 108000s weekly, refresh Sep 19 02:00 CEST. TPU quota untouched (72000s available) - potential avenue if GPU wait becomes the bottleneck (needs torch_xla port, not started).
- Val scripts written and committed for post-refresh queue: realmlp_val_496.py (455f+X24) and realmlp_val_644.py (455f+X23+X24).

## 2026-09-18 13:27 - kfold678c complete (analysis only, contains XS75)
- OOF cos: 61-70 0.172203, 66-70 0.182436, 40-64 0.149658. 15/15 models, 33931s.
- vs kfold530 (v14): +0.0021/+0.0044 - X23 keeps adding on OOF, as in single val (0.1752 vs 0.1734).
- NOT a submission candidate: late OOF 0.1824 < 0.185 threshold and the build contains XS75, whose val/OOF gains were shown to invert on LB (v14 0.135 < v13 0.139). The X23 contribution must be re-measured without XS (x603 val, queued for GPU refresh 02:00 CEST Sep 19).

## 2026-09-18 14:12 - External mining: senanuretin holdout-vs-LB kernel (very relevant)
- senanuretin/why-my-hold-out-and-the-leaderboard-disagreed: (1) price=0 in book = EMPTY LEVEL, not a price - treating zeros as prices flips spread sign and halves mid; (2) market spans 600s vs order/transaction 60s (known); (3) order_action 0=NEW/1=CANCEL confirmed by book balance (matches our X24 schema); (4) monthly target std swings 2.69x - WHICH months you validate on decides the score; his holdout was 83rd-percentile easy; de-bias = reported x (typical/your block); fold-to-fold std 0.0041 means gains <0.005 are at noise floor (matches our val noise experience); (5) cosine is scale-invariant but NOT shift-invariant (don't add constants to preds) and weights rows by ||y|| magnitude - high-vol rows dominate.
- ACTIONABLE: our features22.py and build_x24.py compute mid=(a1+b1)/2 and spread=(a1-b1)/mid WITHOUT masking zero-price rows. If zeros are common, spread/mid/micro features carry systematic garbage. Auditing incidence now (mscapital-audit-zeros CPU kernel; local train/market.feather is truncated - footer corrupt - noted for re-download).
- Cosine weighting implication: our loss DOWNweights |y|>0.001 samples (w=0.5), but cosine weights by ||y|| - high-magnitude rows matter more on the metric. Tension worth a val A/B later.
- bestwater/ktpu-tabm-cos-v6-3seed pulled and is byte-identical to research/reference/bestwater_tabm_v6.py (already mined). pavloivanin EDA new but low votes.

## 2026-09-18 14:58 - X24c built (49 cols) + dataset mscapital-x24c
- Audit result: zero-price rows = train 0.47% / test 0.72% of market rows; 0.56%/0.82% of samples affected. Given typical relative spreads ~1e-3, one unmasked zero-row (relative spread +/-2) dominates a sample's spread mean -> real corruption for affected samples.
- build_x24c: empty-level rows masked out of all price stats; dead-row incidence kept as 8 new liquidity-void features (frac empty ask/bid per bucket, void drift, void asymmetry). Shapes verified: 1257637x49 / 647896x49, no NaNs.
- Val variants ready for GPU refresh: realmlp_val_504.py (455f+X24c), realmlp_val_652.py (455f+X23+X24c). These supersede x496/x644.

## 2026-09-18 16:24 - val A/B queued: x455w (no downweight of high-|y| rows)
- Hypothesis (senanuretin kernel): cosine factors over partitions with weights ||y_g||*||p_g|| - high-magnitude targets dominate the metric. Our loss DOWNweights |y|>0.001 rows (w=0.5), fighting the metric. x455w = champion 455f val recipe with w=1 everywhere. Queued for GPU refresh; compare vs x455 val baseline (0.1671/0.1749).

## 2026-09-19 02:07 - GPU quota refreshed (0s/108000s, next refresh Sep 26), val queue firing
- mscapital-x603-valb (455f+X23 no XS) and mscapital-x504-valb (455f+X24c) RUNNING. x652 and x455w queued behind them (max 2 GPU sessions).
- Mount regression found+fixed: datasets now land under a new /kaggle/input layout, hardcoded /kaggle/input/datasets/<owner>/<slug> paths broke. All val scripts now resolve mounts by filename search (find /kaggle/input -name). CancelKernelSession is 403 via kagglesdk - dead kernels must error out on their own.

## 2026-09-19 03:50 - mount layout fully mapped; x603c + x504c RUNNING
- Probe kernel revealed true layout: datasets at /kaggle/input/datasets/<owner>/<slug>/, kernel outputs at /kaggle/input/notebooks/<owner>/<slug>/. Two bugs found: (1) X23 was passed as a dataset but lives in kernel output mscapital-build-x23 -> silent no-mount; (2) dataset uploads kept the local filename prefix (mscapital-build-x24c_X24c_train.npy) because ApiDatasetNewFile ignores the intended name and upload_file names the blob by basename -> exact-name find missed. Fixed via new dataset version of mscapital-x24c with clean names (X24c_train/test.npy).
- mscapital-x24 dataset (unused now) still has prefixed names - fix if ever needed.
- ~1h of 403s on kernels get/status endpoints (~02:47-03:47) resolved on its own - transient Kaggle auth flakiness; list_kernels kept working throughout.
- Pushed 03:48: mscapital-x603c (455f+X23, kernels=0726+build-x23) and mscapital-x504c (455f+X24c, datasets=+x24c). ETA ~1h each.

## 2026-09-19 05:20 - x603 val 0.169080, x504 val 0.167843: both WITHIN NOISE of x455 (0.1671)
- x603 (455f+X23 no XS): +0.0020 over x455 - below the 0.004 fold-std noise floor. X23 without XS does NOT clearly add; the apparent X23 gain at 678f was riding on XS.
- x504 (455f+X24c): +0.0007 - flat.
- Consequence: no kfold submission justified by these. v13 (455f) remains champion at LB 0.139. The val ceiling around 0.167-0.175 has not moved without the LB-toxic XS layer.
- x652c (455f+X23+X24c) and x455w (no downweight A/B) launched 05:20.
- Open direction: gains must come from something orthogonal - better targets/loss (455w test), post-hoc calibration, or a structurally different feature family.

## 2026-09-19 06:06 - x652 0.168738, x455w 0.166260: both flat. 455f stack at local ceiling.
- x652 (455f+X23+X24c): no interaction. x455w (w=1, no downweight of big targets): 0.1663 vs 0.1671 - no gain, hypothesis dead (or neutral). Four post-quota vals all within noise of x455.
- Standing: v13 LB 0.139 champion; no new submission candidate; feature adds X23/X24c closed as flat-without-XS.
- Next lever launched: x455cap (N_ENS 24, EPOCHS 12) capacity A/B on the champion recipe.

## 2026-09-19 08:20 - Mining yunsuxiaozi/rfmf-realmlp (the public kernel our recipe descends from)
- Differences vs our champion: (1) they split by ROW ORDER (first 800k samples) not by month; (2) they drop |corr|>=0.9 pairs + constant columns from the 0726 set (~100 cols dropped) - we keep all 152; (3) they quantile-bin high-cardinality numerics into categoricals and feed a categorical embedding layer alongside numerics; (4) RQ aux target (we dropped it, tested negative by UnseenAnchor); (5) y rounded to 4 decimals.
- Our numeric-only 455f beats their public LB, but two untested-by-us elements remain: correlation pruning and binned-categorical embeddings.
- Queued: realmlp_val_455corr.py (corr-prune |r|>=0.9 fit on train<=60). Binned-cat embeddings parked (bigger change, lower prior).

## 2026-09-19 09:04 - x472 0.167634, x455corr 0.165400: flat again. 7/7 flat vals.
- x472 (455f+X25 trade arrival/impact): +0.0005. Third feature family (X23, X24c, X25) flat without XS.
- x455corr (|corr|>=0.9 prune of 0726): -0.0017 vs x455. Correlation pruning closed (neutral-to-negative).
- The 455f recipe is robustly at its local ceiling: 7 consecutive controlled vals within +/-0.002 of 0.1671.
- Pivot: (1) kfold455 5-seed re-run saving per-model OOF preds -> enables post-hoc shrinkage calibration + slightly better averaging; (2) keep mining for structural ideas.

## 2026-09-19 09:04 - kfold455s5 launched (parent-approved)
- 5 seeds x 5 folds = 25 models, saves oof_models_kfold455s5.npy (25 x 1.26M) + test_models (25 x 648k) for offline shrinkage calibration. ETA ~13h (~22:15 CEST). Uses ~47k of 108k GPU seconds.

## 2026-09-19 10:57 - x603 seed replication: 0.165685 (seed42) vs 0.169080 (seed2026) - X23 line CLOSED
- Same config, two seeds: 0.0034 spread. The x603 +0.002 "gain" over x455 (0.1671) was seed noise. Seed-variance range ~0.003-0.004 matches the senanuretin fold-std noise floor.
- X23 does not add without XS. kfold603 is dead as v15 path. 8/8 controlled vals flat.
- Meta-lesson confirmed with own numbers: single-val deltas <0.004 are uninterpretable; only kfold-OOF or LB can arbitrate deltas of that size.

## 2026-09-19 20:09 - kfold455s5 KILLED by 10h session timeout at 22/25 models; v2 relaunched
- v1 (pushed 09:04, timeout 36000s) hit the session timeout during seed 99 fold 4. Saves were end-of-run only, so ALL output lost (~39.6k GPU s burned, 0 artifacts). My design flaw, logged.
- Fix (realmlp_kfold455s5v2.py): incremental saves after every model (oof_models_*_partial.npy, test_models_*_partial.npy, metrics_*_partial.json), FOLD_IDX env filter, timeout 43200s (12h; projected need ~39.2k s).
- Relaunched 20:09 as diegoaranguren/mscapital-kfold455-5seed-v2-incremental-save (RUNNING). ETA ~07:10 CEST Sep 20. GPU quota after v1: 53.0k s left this week; v2 uses ~39k, leaves ~14k.
- If v2 also times out: partials survive; a complementary FOLD_IDX run completes the matrix.

## 2026-09-20 10:10 - kfold455s5v2 COMPLETE: 25/25 models, OOF 0.16827/0.17522; calibration lines tested and CLOSED
- Run: 42,160s (11.7h, just under the 12h timeout). OOF 61-70: 0.168274 (v13 kfold455: 0.1678, +0.0005); 66-70: 0.175222 (v13: 0.1745, +0.0007); 40-64: 0.147749 (month-difficulty skew confirmed again).
- Per-fold val cos 0.142-0.172 across seeds; seed spread within folds ~0.003 (consistent with known noise floor).
- Shrinkage grid (per-row std across 25 OOF preds, weight = 1/(1+lam*sn_norm)): ALL lambdas HURT (-0.002 at lam0.25 to -0.035 at lam4.0). Rank-based top-q shrink also hurts. DEAD: model disagreement is signal, not noise.
- Anti-shrinkage (amplify high-disagreement rows): catastrophic (-0.09 at lam-0.5). Confirms direction.
- Robust aggregation: median -0.0002, trimmed mean -0.0009 vs plain mean. DEAD.
- Greedy forward model selection (select on months 40-64, eval 61-70/66-70): all-25 equal mean is best; selection does not transfer. DEAD.
- Conclusion: s5 value = variance reduction of the exact v13 recipe (25 vs 15 models). OOF delta +0.0005/+0.0007 - inside noise floor but structurally >= 0 by construction. Submission decision escalated to parent (his pre-set condition "OOF supera a v13" is literally met on both windows; margin is inside noise).
- Artifacts: oof_models_kfold455s5.npy (25x1.26M), test_models_kfold455s5.npy (25x648k), metrics_kfold455s5.json in /tmp/work; kernel output at diegoaranguren/mscapital-kfold455-5seed-v2-incremental-save.

## 2026-09-20 10:14 - v15 (kfold455s5) LB 0.138: FLAT vs v13 0.139. LB noise resolution confirmed ~0.001
- v15 submitted (ref 56386336): 25-model simple mean, exact v13 post-processing. Public LB 0.138 vs v13 0.139 - no gain, Kaggle keeps v13 as champion (rank 125/225, 20 entries).
- OOF said +0.0005; LB said -0.001. Confirms: OOF deltas under ~0.002 do not resolve on LB. Only structural changes move the board.
- Board movement: 225 teams (was 223). Top1 涵哥 0.176 (was 0.172, +0.004 jump - someone found something real). gate10 0.161 (was 0.160). Diego dropped 118->125 purely from other teams passing 0.139.
- Calibration lesson (final): with a 25-model OOF tensor in hand, every post-hoc calibration tried (shrinkage, anti-shrinkage, median, trimmed, greedy selection) was neutral-to-harmful. Equal-weight mean of a well-seeded k-fold ensemble is the ceiling for this recipe. Future gains must come from features/architecture, not aggregation.

## 2026-09-20 11:21 - X26 build launched (CPU kernel) + open-source single-model options
- build_x26.py pushed as diegoaranguren/mscapital-build-x26-ofi-vpin-volume-clock (CPU). Features: CKS OFI (L1+L2, valid-masked), OFI accel, Roll spread, Amihud, microprice tail, VPIN K=10/20 (volume-clock, midpoint-bucket rule - chunk-carry vectorization unit-tested against exact reference).
- Formula validation: stefan-jansen ML4T ch.8 (Amihud |r|/dollarvol, Roll 2*sqrt(-cov), OFI trade-based) - mine match standard definitions; CKS OFI per Cont-Kukanov-Stoikov.
- If X26 val (455f+X26) < +0.004: feature engineering CLOSED. Next lever per iamltpn (5th, public): single-model training. Open-source candidates: (a) RealMLP hyperparameter random search (lr/epochs/n_ens/hidden, ~1.5k GPU s per config, 5-6 configs fit the weekly quota) - highest prior, champion arch has never been tuned, only carried; (b) ModernNCA (public, instance-based inductive bias, memory-heavy on 1.26M rows - risky on P100); (c) FT-Transformer tabular (low prior: bestwater's Transformer LB-negative).

## 2026-09-20 14:20 - X26 (OFI/VPIN volume-clock, 469f) FLAT: val 0.16797 vs champion 0.1671 (+0.0009, gate +0.004 NOT cleared)
- Kernel diegoaranguren/x26val469 (slugs mscapital-val-x26-469f/-v2 bricked server-side 403/409; fresh short slug works).
- Panel on saved val preds: overall 0.1674, no66 0.1482, recent 0.1477, worst month 70 (0.1263), monthly std 0.0122. X26 adds nothing beyond noise.
- DECISION per parent gate: feature engineering CLOSED. Fourth feature family flat (X23, X24c, X25, X26) without XS. Aggregation ceiling reached earlier -> next axis = RealMLP hyperparameter random search (parent-authorized fallback, Sep 20 11:18).
- 14:20 GPU quota: 6404 s left (resets Sep 26 02:00 CEST). TPU 72k s untouched -> TPU port spike started.
- 14:25 random search launched: rs1 (N_ENS8 EP7 LR2e-3 BS256), rs2 (N_ENS8 EP7 LR5e-4 BS512) running; rs3 (N_ENS8 EP12 LR1e-3 BS256) queued (max 2 concurrent GPU batch sessions). ~1.5-2k s each, leaves ~2k s buffer.

## 2026-09-20 16:20 - TPU UNLOCKED (TpuV5E8 + python-tpuvm docker image)
- Root cause of spikes 1-3 failing: default image lacks torch_xla AND machine_shape must be TpuV5E8 with docker_image gcr.io/kaggle-private-byod/python-tpuvm@sha256:a2111cb9... (metadata mined from bestwater/ktpu-tabm-cos-v6-3seed via API; his kernel source + kgpu-tabm-cos689 saved to /tmp/work).
- spike4: torch 2.8.0+cpu + torch_xla 2.8.0, device xla:0, training works. TPU quota 72k s = 20h.
- realmlp_tpu.py (realmlp_rs + XLA device/step) launched as diegoaranguren/tpuval455: exact champion replication (455f, N_ENS16, EP9, BS256, LR1e-3). Validates port if val cos ~= 0.1671. Then path to kfold455x5seed on TPU.
- bestwater reference numbers (his kernel docstrings): GPU TabM cos689 raw pool=0.16379, raw last=0.18697; TPU v6 matches GPU config, 3 seeds ~50-60 min on v5e-8.
- rs1/rs2 still API-wedged 403 but consuming GPU quota (~1.6k s gone, 4.7k left); results only via cloud browser session logs later.

## 2026-09-20 18:20 - Root cause of wedged kernels: slug != title
- Kernels pushed with title-derived ref != slug wedge server-side: 403 on every session endpoint (status/output/cancel/delete), zombie sessions. Wedged: val-x26-469f v1+v2, rs1, rs2 (all had title != slug). Working ones had title == slug (x26val469, tpu-spike*, builds).
- FIX: kaggle_io.push_kernel now forces new_title = slug.
- rs1/rs2 zombie GPU sessions can't be cancelled/deleted; they auto-timeout (12h max). GPU batch slots may be held meanwhile. rs random search ABANDONED (results unreachable, rerun costs ~3.3k of the 4.7k s GPU left; marginal expected gain).
- tpuval455b (title==slug) RUNNING and readable. TPU port validation in flight.
- Vault Kaggle password stale (invalidLogin); vault request sent to Diego via parent/WhatsApp.

## 2026-09-20 18:40 - Random search CLOSED (flat); TPU v1 failure root-caused
- rs1 (N_ENS8 EP7 LR2e-3 BS256): best val 0.16659 @ ep6 - within noise of champion 0.1671, flat.
- rs2 (N_ENS8 EP7 LR5e-4 BS512): 0.15786, still climbing at ep7 - undertrained, worse.
- Random search conclusion: champion recipe (N_ENS16 EP9 LR1e-3 BS256) is locally optimal; no hyperparam gain available at search resolution. CLOSED.
- tpuval455 v1 FAILED at 2617s: XLA "LLVM compilation error: Cannot allocate memory" -> segfault. Cause: per-step LR updates (flat_anneal per step) invalidate the compiled graph every step -> recompilation memory bomb. bestwater's TPU code updates lr per EPOCH (CosineAnnealingLR T_max=EPOCHS). Also its Logs panel showed Accelerator: None - CPU-vs-TPU unconfirmed; HW probes added.
- Fix: realmlp_tpu.py now per-epoch LR + hardware probes (get_memory_info, xr device count, jax.devices). tpuval455c pushed (title==slug). tpuval455b (old code) still running as CPU/TPU data point.
- Kaggle web login recovered (Diego updated vault password). Wedged-kernel results read via browser Logs tab.

## 2026-09-20 19:20 - TPU BLOCKED: Kaggle requires Persona identity verification for TPU v5e-8
- Editor UI accelerator options on Diego's account: None / GPU T4 x2 / TPU v5e-8 (listed). Selecting TPU triggers: "Scarce resources such as TPU v5e-8 require identity verification... verification with a smartphone or webcam on Persona (3rd-party)". Screenshot: /downloads/cloud-browser-20260920-171912.png.
- Account NOT verified -> no TPU until Diego completes Persona verification (his ID/selfie, ~5 min, only he can do it).
- This also explains: API machine_shape/enable_tpu/create_kernel_session all silently ignored; all "TPU" sessions ran on CPU (Accelerator: None; jax CpuDevice; HW world 1 1). spike4's xla:0 was XLA:CPU.
- tpuval455b crashed OOM at ~2617s (per-step lr recompile bomb on CPU XLA). tpuval455c (per-epoch lr) still running on CPU - may complete and validate the port numerics anyway.
- Compute now: GPU 4.7k s until Sep 26 02:00 CEST reset. TPU 72k s waits on Diego's verification.
- 20:10 tpuval455c also OOM (LLVM compile, 2255s) despite per-epoch lr: every lr change forces a fresh XLA graph compile and host RAM accumulates executables. CPU-XLA port validation ABANDONED. TPU path parked until Diego's Persona verification. GPU 4.7k s preserved for submission-path runs.

## Sep 20 21:20 — TabM@455f flat → línea cerrada; blend marginal
- TabM (arch bestwater reimplementada, Apache 2.0) @455f, folds 60-64/65-70, seed 2026: fold4 0.14442, fold5 0.15720, OOF 61-70 0.15206. Bar 0.165 → standalone muerto. La ventaja de bestwater (pool 0.1638 @689f) viene de sus 462 features privadas, no de la arquitectura.
- Blend check (RealMLP 25-model OOF + TabM OOF): preds TabM con std 114× la de RealMLP (cosine loss sobre target raw) → z-score obligatorio. Blend z-scored: +0.0013 OOF con peso tuneado en la misma ventana (overfit); forward-sim honesto (w fit 61-64 → eval 65-70): +0.0010 (0.17392 vs 0.17291), 66-70 +0.0009. Correlación r,t = 0.83 → poca diversidad.
- Decisión: +0.001 < resolución LB (regla 0.002) y < ruido val → NO es evidencia fuerte para v16. Línea TabM parada y documentada. Test preds de TabM guardadas por si un futuro v16 las quiere de diversidad barata, pero no justifican submission por sí solas.
- X27 (depth-cross) depriorizado: EXP030 de ZWQ1037 falló -0.0011 sobre TabM fuerte (features de árbol solo ayudan a baselines débiles).
- Estado cómputo: GPU ~3.2-3.5k s hasta reset Sep 26 (reservada para submission path); TPU 72k s intacta pero bloqueada en verificación Persona (pendiente Diego).

## Sep 20 22:20 — Research: loss-balance es la palanca no testeada; X28a lanzado
- TPU probe v2 (tpu-probe-persona): jax CpuDevice, sin accelerator env → Persona sigue pendiente (22:15). TPU bloqueado.
- Deep-read toolkit ZWQ1037 + notebook público yunsuxiaozi/rfmf-realmlp (37KB código parseado de ipynb). Nuestro campeón YA es la familia Yunsu: PBLD embeddings, NTPLinear ensemble n_ens=16, ScalingLayer, EMA 0.998, weighted MSE (0.5 si |y|>0.001). Deltas restantes: (1) balance de loss — nosotros lambda_cos=1.0 (coseno domina el gradiente), Yunsu 0.01; ablación ZWQ1037 EXP_REALMLP_001_003 en 379f: 4 miembros + weighted MSE + 0.01 cos = 0.1371/0.1449 vs MSE plano 0.1283/0.1396 (+0.008/+0.005); (2) cabeza auxiliar RQ-KMeans (4 capas × 5 códigos, lambda 0.1) — prioridad declarada de ZWQ, sin testear por él ni por nosotros; (3) n_ens 16 no mejoró estable en su ablación.
- bestwater TabM confirmado en su doc de arquitectura: 64 cabezas promediadas ANTES del loss = matemáticamente un solo head lineal; su ventaja = features privadas. Cierra definitivamente la línea TabM.
- YangQ fusion pack (lb0142): 60% MultiStream CNN-Transformer + 40% RealMLP 8 miembros; su RealMLP también es familia PBLD → convergencia de evidencia: el camino es afinar nuestro single model, no más fusión.
- X28a LANZADO (22:16): kernel diegoaranguren/mscapital-x28a-lc01 — campeón exacto salvo lambda_cos 1.0→0.1, seed 2026, solo fold5 (65-70). Comparación: campeón fold5 seed2026 = 0.17009 (1821s). Gate señal: +0.004 (ruido val). Coste ~1.8k s de los ~3.3k GPU restantes; deja ~1.5k buffer. Script: src/kaggle/realmlp_x28a_lc01.py.
- Si señal: confirmar fold4 + panel completo tras reset GPU (Sep 26) o en TPU si Persona; luego RQ aux (X28b). Licencia notebook Yunsu: None → reimplementación de ideas, no copia literal.

## Sep 20 23:15 — Sandbox wipe recuperado; X28a v1 falló por data source, v2 lanzado
- Wipe de /tmp entre 22:16 y 23:11: repo re-clonado desde GitHub (commits intactos), creds Kaggle restauradas (~/.kaggle/access_token), kagglesdk reinstalado.
- X28a v1 ERROR a los 35s: AssertionError '0726 not found' — el script fusiona las 152 features públicas del kernel-data-source yunsuxiaozi/rfmf-0726data y no lo adjunté. Consumo ~35s GPU solamente.
- X28a v2 lanzado (23:13) con kernels=('yunsuxiaozi/rfmf-0726data',). ETA ~23:45.
- Lección: los kernels campeones llevan TRES datasets + UN kernel data source (rfmf-0726data); documentado para futuros pushes.

## Sep 21 00:20 — X28a NEGATIVO: lambda_cos 0.1 = 0.168294 vs campeón 1.0 = 0.170090 (fold5 seed2026)
- Log del kernel (web, línea "seed 2026 fold 5: best val cos 0.168294 (1933s)"): -0.0018 vs campeón. Dirección negativa, dentro de ruido pero SIN señal → palanca loss-balance CERRADA: lambda_cos=1.0 del campeón es óptimo local; no probar 0.01 (dirección opuesta empeora).
- La evidencia de ZWQ1037 (0.01 > MSE plano en su 379f/4-miembros) NO transfiere a nuestro 455f/16-miembros: régimen distinto. Lección: sus ablaciones orientan hipótesis pero requieren validación propia.
- Crash post-training (línea 308, sample_submission no encontrado — sin competition data adjunta): solo afecta al CSV final, métricas ya registradas en log. Kernel errored no expone ficheros vía API.
- Nota: el log muestra "(1257637, 450)" — el build real produce 450 columnas, no 455 (el tag 455f es histórico). Verificar contra build campeón antes de documentar el número en README.
- GPU: ~1.6k s restantes tras v2 (2165s) → bajo el mínimo para otra pantalla (1.9k). Sin más runs GPU hasta reset Sep 26 00:00 UTC.
- Infra: wipe de sandbox ~23:00 + caída del proveedor 00:13-00:18 (admission full) — recuperado todo, sin pérdida (repo en GitHub, creds en prompts).
- X28b (RQ aux head) queda encolado para post-reset o TPU. TPU: Persona sigue pendiente.
- Decisión sobre datos locales: /tmp/mscapital (9GB) no se reconstruye de momento — el trabajo actual es kernel-side; arrays pequeños (full_y/full_month) se bajarán del dataset mscapital-matrices cuando hagan falta para paneles.

## Sep 21 01:15 — X28b listo; investigación: la meseta pública es real
- X28b construido (src/kaggle/realmlp_x28b_rq.py): campeón exacto (lambda_cos=1.0, ruido de label, EMA 0.998) + cabeza auxiliar RQ-KMeans (config real de yunsu: 3 capas × 3 códigos, NO los defaults 4×5; códigos sobre y limpio redondeado a 4 dec, lambda_rq 0.1). Listo para lanzar tras reset GPU o en TPU.
- Resuelto lo de 450 vs 455: el campeón es 298f advdrop (303-5) + 152 públicas = 450 columnas reales; el tag "455f" es histórico. Header del script ya lo decía. Pendiente corregir README (dice 455).
- Toolkit ZWQ1037, informe goal_improve_beyond152: su objetivo "superar 0.152 con ≤3 submissions" CERRÓ SIN MEJORA VERIFICADA. Su rama más fuerte (market-conditioned event residual sobre Transformer) dio forward delta +0.0013 — meses de trabajo de modelos de secuencia para +0.001. Conclusión: todos los que minan datos públicos están en meseta 0.15-0.16; el diferenciador de bestwater son sus 462 features privadas. Nuestro techo con material público está cerca; prioridad = robustez del campeón (más seeds/folds vía TPU) + micro-ganancias verificadas (RQ aux) + disciplina de validación.
- Su protocolo de validación es más estricto que el nuestro: train 0-59, purge 60-61, select en 62-65, confirmación forward 67-70 (excluye 66). Adoptar para cualquier candidato v16: selección y forward-confirm en ventanas disjuntas.
- TPU probe v3 lanzado 01:13; resultado pendiente.
