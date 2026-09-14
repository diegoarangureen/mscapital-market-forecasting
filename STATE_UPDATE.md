# 2026-09-12 18:30 CEST - wipe #3 recovery (new agent)
- Sandbox wiped again at ~18:25; /tmp lost incl. running robustness check (eval_shift GBM part).
- Recovered ALL scripts from transcripts (this dir), repo recloned, data re-downloading.
- Candidate awaiting robustness+submission: 50/50 unit-blend of GBM v4 + MLP(121 feats: X2 + fp log-diff/log1p + X6).
  Val (61-70): blend 0.13126 centered (GBM 0.126436 recon / MLP 0.125204). Weight plateau smooth (0.4: 0.13115, 0.5: 0.13126, 0.6: 0.13090). corr(preds)=0.84.
- Plan: rebuild X2/X6/fp -> eval_x2base (expect 0.1258/0.1264) -> mlp.py (expect ~0.1313 blend) -> eval_shift.py + mlp_shift.py (robustness train0-50/val51-60) -> if holds, refit 0-70 both, blend, submit "GBM v4 + MLP blend".
- Submission quota: ~5/day, reset ~02:00 CEST.

## 2026-09-13 20:27 CEST - rebuilt + reproduced + robustness passed
- Rebuild: X2/X6/fp all rebuilt; baseline 3-seed 0.12469/0.12633 centered (matches predecessor 0.12644).
- Blend reproduced: MLP seedavg 0.12506 centered, corr 0.8373, blend w=0.5 val 0.131141 centered (smooth plateau).
- ROBUSTNESS (train 0-50, val 51-60): GBM 0.122894 centered, MLP 0.116482, blend 50/50 = 0.124462. Blend beats GBM on BOTH splits (+0.0048 main, +0.0016 shift). GO for submission.
- Running: mlp_epochs.py -> /tmp/work/mlp_best_epochs.json (log /tmp/mlp_epochs.log, marker EPOCHS_DONE).
- NEXT: run final_blend.py (GBM refit 0-70 @1.1x iters [110,116,77] + MLP refit @1.1x best epochs + 50/50 unit blend + nodata zero + clip + submission_blend.csv; marker FINAL_BLEND_DONE, log /tmp/final.log). Then submit via submit.py pattern (fix path to submission_blend.csv, description "GBM v4 + MLP blend"). Check quota first via list_submissions. Then leaderboard screenshot via cloud browser and report LB+rank to parent.

## 2026-09-13 21:07 CEST - SUBMITTED blend v1
- ref 56213569, public LB 0.113 (previous best 0.110, GBM v4). Description "GBM v4 + MLP blend".
- Refit preds saved: refit_gbm_test.npy, refit_mlp_test.npy (in src/recovered/). mlp_best_epochs {0:8,1:9,2:15}, x2 iters [110,116,77].
- Val->LB mapping: val 0.1274 -> LB 0.110; val 0.1311 -> LB 0.113. Gap remains large; next levers must close val-LB transfer.

## 2026-09-13 21:45 - structural finding: test regime = late months
- Per-month val cosine flat 0.10-0.17 except month 70 (hardest, 0.097-0.107). LB 0.113 ~= month-70 level: test (months 71+) follows the late regime. Late val (66-70) is a better selection metric; blend w=0.5 optimal on both (late 0.13542).
- Feature drift PSI train->test: activity/count features drift hardest (ord_nnew 0.130, ord_ncan 0.126, mk_cnt_total 0.125, tx_n 0.095).
- Queue: E3 log1p count features (running, /tmp/e3.log), E2 recency ramp re-scored on late val (eval_recency2.py), then cohort GBM, 1D-CNN on bar series.

## Sept 14 00:00-05:00 — Round 3: systematic pivot attempts (all dead ends except pseudo-sims)
- Seq GRU (60x6 bar seqs, 150K-200K subset): NEGATIVE signal (seedavg -0.017; blend worsens). Family discarded.
- Month-cohort GBM (train on 31/41-60): monotone worse (0.1236→0.1126). More data always wins.
- Drift-feature ablation (drop top-5 PSI): hurts late val (0.1266→0.1243). Drifting features carry real signal.
- Pseudo-labeling (self-training on forward months, w=0.1, confident half): POSITIVE in 2/3 forward sims (+0.0023 late on eval61-70, +0.0009 on eval66-70, neutral on 51-60). Applied to test: v2 LB 0.113 (=v1), v3 (GBM+MLP both pseudo) LB 0.112. LB-neutral at 3-decimal resolution.
- Adversarial importance weighting: AUC 0.74 (strong shift train-vs-forward) but weighted training hurts (0.1186 vs 0.1216). Dead.
- Ridge 3rd model: hurts blend at any weight. Dead.
- Blend weight sweep: flat 0.45-0.50. No lever.
- Diverse-config GBM (B_deep/C_shallow): blend +0.0002 full = noise. Dead.
- Big MLP (512-256-128): 0.1158 vs small 0.1251. Overfits. Dead.
- Rank-target GBM: solo 0.085, blend hurts. Dead.
- Fingerprint-view GBM: solo 0.040, blend hurts. Dead.
- 60-grid fingerprints (10s) for MLP: 0.1155 vs 30-grid 0.125. Noisier cells. Dead.
- LB probes (refs 56217171/56217183): blend 0.113 > GBM-only 0.110 > MLP-only 0.109. Diversity transfers; late-val ranking does NOT rank close variants on LB.
- Structural: month-70 blend cos 0.107 ≈ LB 0.113 → LB measures late regime. No temporal leak in test (sbp 0..597 all pre-predict).
- CHAMPION unchanged: GBM(58)+MLP(121) 50/50 unit blend, refit 0-70, LB 0.113, rank ~198.
- Box note: nan_to_num on huge arrays OOMs (3 full bool masks) — always blockwise. glibc heap creep needs malloc_trim; background jobs >1.2GB spike-killed, prefer foreground for builds.
