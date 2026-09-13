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
