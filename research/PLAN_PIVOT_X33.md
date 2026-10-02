# X33 pivot plan (draft for Diego's green light, 2026-10-02)

Status: PLAN ONLY. No compute launched. No submission. Diego's 2 Oct 16:12-16:13
WhatsApp: "Haz el entrenamiento y súbelo" -> option 1 (pivot).
Rules in force: TRAINING_AGENT.md protocol, promotion gate (+0.002 stable vs
control, same metric/protocol, both folds, both seeds 2026/42, not one month),
public material only (host bans external data/models, so no pretrained MarS etc.),
solo entry, inform Diego before any submission.

## Where we stand (facts from the repo log)
- Champion flow31 v1 = public LB 0.141. Top-10 cut 0.167 (as of Sep 28): gap +0.026.
- Closed lines: GRU streams x3, TabM (corr 0.83 with RealMLP, blend +0.001),
  RQ-KMeans aux, quantile scaling, loss-balance, XS features (3 LB failures),
  X32 ctx/per features (seed 42 flipped seed 2026 sign).
- Seed noise floor ~0.003-0.004 on one fold. Any screen below +0.002 is not evidence.
- Public ceiling per other teams' closeouts is ~0.15-0.16 on public data/models
  (research/LOG.md, ZWQ1037 closeout). The rank-1 team says competition data only
  (discussion/742707), so headroom exists but nobody has published the lever.

Honest expectation: a passing gate means about +0.002, i.e. ~0.143. This plan does
not promise top-10.

## Candidates

### A (recommended, cheap first): target geometry for the cosine objective
Hypothesis: the cosine loss and the raw-y scoring are dominated by a few
heavy-tail windows. Train on a per-fold winsorized target (clip y at the train-fold
q99.5 / q0.5, loss only; score still uses raw y), keep everything else identical.
Never tested in the log (searched winsor/huber/target clip: nothing).
Evidence level: hypothesis. Related public hint: bestwater rank-normalizes the
target per month for CV (R1_METHODS_LANDSCAPE.md), which is a different
transform; the rank variant would be arm A2 only if A1 shows any sign.
Cost: config + loss-target change in the training script. No data rebuild.

### B (conditional on a free CPU probe): multi-horizon self-supervised aux heads
Source: Xu Dian, discussion/742719 (unvalidated claim that the label contains a
term from ~90-120 s ahead), and the multi-horizon LOB literature
(arXiv 2105.10430, multi-horizon encoder-decoder on LOB; Kolm et al., "Deep order
flow imbalance", Mathematical Finance). Idea: add aux heads that predict
in-window returns at several horizons from earlier parts of the window, with the
aux target segment excluded from the inputs (leak check required).
Phase 0 probe (CPU, 0 GPU/TPU): check whether in-window return at horizons
10-120 s is predictable from earlier-window features and whether that predictability
peaks where the claim says, using existing prepared data. Run B on GPU only if the
probe shows a clear peak (rank IC well above noise on 2+ time blocks).
Cost if it passes: new target builder (CPU) + aux head in the trainer.

### C (not recommended): sequence-model family as ensemble member
Public reference: zwq1037 factorized transformer (LB 0.145 single). Reasons against:
our GRU streams died three times, and ZWQ's own closeout found only ~+0.001 as a
blend member. High build cost (raw stream tensors), TPU queue risk (4 TPU screens
died in queue). Keep as last resort.

## Phases, caps and gates (current quota: GPU ~10.8 h, TPU ~8.5 h; refresh 3 Oct)
Per-model cost on GPU is ~0.5 h (v3: 4 models in 1.98 h).

| Phase | What | Backend | Cap | Gate to continue |
|---|---|---|---|---|
| 0 | Repo recovery, B probe, A code + unit tests (CPU) | CPU | 0 GPU/TPU | probe result decides B |
| 1 | Screen: fresh flow31 control + arm A1 (+ B if probe passes), folds 0 and 4, seeds 2026/42, 4 models per arm | GPU | 6.0 h (control 2.0 + A 2.0 + B 2.0); hard stop 6.5 h | per arm: +0.002 stable vs control on f0 AND f4 AND both seeds, monthly breakdown not one-month, two-seed ensemble delta positive. Else the arm closes. |
| 2 | Temporal confirmation of the survivor (control vs candidate, 2 seeds x 5 folds x 2 origins) | TPU preferred (GPU fallback with reduced panel) | up to 12 h, only after the 3 Oct refresh | gate +0.002 in both blocks, bootstrap CI reported, all deltas kept |
| 3 | Final: 15 models (5 folds x 3 seeds, origin 70) | GPU | 8 h | none; produces CSV only |
| 4 | Submission | Kaggle | 1 slot | needs Diego's explicit OK on a submissions strategy first |

Control reuse: any change to the trainer/dataset code forces a fresh control
(Diego's rule), hence the control in the Phase 1 budget.
Timeline vs deadline (9 Oct 16:00 UTC): Phase 0 tonight, Phase 1 on 3 Oct,
Phase 2 4-6 Oct, Phase 3 7 Oct, submission decision 8 Oct.
If no arm passes Phase 1: stop, keep 0.141, report. Retraining the 0.141 recipe
(option 2) stays available but is the measured ceiling.

## Open items needing Diego/Main
- Green light on this plan before Phase 1 launches (Phase 0 is CPU-only and
  starts now with no compute).
- Env was rebuilt: clone is public/read-only, no push credential, no Kaggle token.
  Kaggle token regenerates via the vault browser flow. A GitHub PAT may need a
  sudo email code from Diego.
