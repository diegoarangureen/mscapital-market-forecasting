# v16 candidate plan (draft, Sep 21 2026) — NO submission without explicit owner authorization

Champion: v13 = LB 0.139 (rank ~125/225). v15 (25-model) = 0.138 flat.
OOF reference (kfold450 5-seed v2): 61-70 = 0.16827, 66-70 = 0.17522. Fold5 seed2026 = 0.17009.

## Hard gates (parent rules)
- val >= 0.130 clear AND robustness verified.
- No submissions on changes below LB resolution (~0.002) unless strong forward-sim evidence.
- Quota: 5 submissions/day, resets ~02:00 CEST.
- Every submission requires explicit owner/parent authorization. Never self-authorize.

## Validation protocol for any v16 candidate (adopted from ZWQ1037, stricter than before)
- Selection window and forward-confirmation window must be DISJOINT:
  select on 62-65, confirm on 67-70, exclude month 66 (anomalous) from scoring.
- Gate: +0.004 vs champion same-window OOF (val noise floor), plus panel:
  all primary deltas >= 0, LOMO min >= -0.001 (src/kaggle/panel.py).

## Candidate paths, ranked
1. **TPU replication + seed extension** (blocked: Persona verification).
   First run: replication SEEDS=2026 FOLD_IDX=4, target ~= 0.17009 fold5 (recipe
   fidelity check of the torch_xla port; per-epoch LR/noise anneal is the only
   intentional difference). Then full 5-fold x 5-seed on TPU (72k s budget,
   ~zero marginal cost vs GPU). v16 candidate = 10-seed mean (50 models):
   expected gain small but robustness-verifiable via OOF tensor.
2. **X28b (RQ-KMeans aux head)** — script ready (src/kaggle/realmlp_x28b_rq.py).
   Screen: seed 2026 fold5, needs ~1.9k s GPU (post-reset Sep 26) or TPU.
   Gate: +0.004 over 0.17009. If signal: fold4 confirm + full panel before any
   submission talk. Nobody public has tested this lever.
3. **Proprietary order-flow features v2** — only if 1-2 stall AND compute unlocks.
   Strict forward-sim gating; the EXP026 lesson (+0.0011 local, -0.015 LB).

## Non-candidates (closed, evidence in LOG/EXPERIMENTS)
- More equal-weight aggregation (v15 ceiling), loss balance (X28a), TabM arch,
  XS/transductive features (v14 + 2 independent confirmations), new public-data
  feature families (X23-X26, RS1/RS2, EXP067/068).
