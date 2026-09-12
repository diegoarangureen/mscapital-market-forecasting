# 2026-09-12 18:30 CEST - wipe #3 recovery (new agent)
- Sandbox wiped again at ~18:25; /tmp lost incl. running robustness check (eval_shift GBM part).
- Recovered ALL scripts from transcripts (this dir), repo recloned, data re-downloading.
- Candidate awaiting robustness+submission: 50/50 unit-blend of GBM v4 + MLP(121 feats: X2 + fp log-diff/log1p + X6).
  Val (61-70): blend 0.13126 centered (GBM 0.126436 recon / MLP 0.125204). Weight plateau smooth (0.4: 0.13115, 0.5: 0.13126, 0.6: 0.13090). corr(preds)=0.84.
- Plan: rebuild X2/X6/fp -> eval_x2base (expect 0.1258/0.1264) -> mlp.py (expect ~0.1313 blend) -> eval_shift.py + mlp_shift.py (robustness train0-50/val51-60) -> if holds, refit 0-70 both, blend, submit "GBM v4 + MLP blend".
- Submission quota: ~5/day, reset ~02:00 CEST.
