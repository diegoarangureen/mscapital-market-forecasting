# X34 training hypotheses — 2026-10-03

Diego authorized training the proposed hypotheses, continuing the existing
project and GitHub updates. Champion remains flow31 no-clipping, public LB 0.142.

## Stage 1: angular loss against the clean target

Control: original noisy-target Pearson angular loss over all internal members.
H1: only its angular target becomes clean. Weighted MSE still uses the same
noisy targets, noisy-target-derived weights, noise realizations and schedule.
Keep architecture, scaler, optimizer, EMA, batch size, epochs and early stopping.
Score on raw competition targets with uncentered cosine, no clipping and the
existing no-data zeroing. This is distinct from the previous clean-MSE-weights
experiment and the failed winsorized-target experiment.

Eight models: fresh control and H1, folds 0/4, seeds 2026/42. Interleave each
fold/seed's control and candidate. Same source/runtime/data; separate immutable
run directories. No checkpoint or predictions from a changed recipe are reused.

TPU kernel: https://www.kaggle.com/code/diegoaranguren/mscapital-x34-h1-tpu
Soft stop: 3.2 h; platform timeout: 12,600 seconds (3.5 h). No automatic retries
or GPU fallback. The earlier 2.23 h TPU run plus this maximum remain below the
previously authorized 8.5 h TPU budget; no GPU work is launched here. Kaggle's
weekly reset is not treated as an instruction to spend the full renewed quota.

## Fixed decision rule

Compare complete eight-model results only. Require delta >= +0.002 in all four
fold/seed pairs and both fold-level seed ensembles, a majority of monthly deltas
positive, and every leave-one-month-out delta positive. Record every result,
including negatives. No cutoff/seed selection after observing performance.

These screen scores use checkpoint-selection windows. Passing the screen earns
temporal confirmation; it does not establish independent generalization or
authorize a submission. Do not loosen the gate after seeing results.

## Other hypotheses

H2 is implemented and its config is staged, but NOT launched: apply angular loss
to the internal ensemble mean while retaining the original noisy angular target
and per-member MSE. Its control can use this exact code snapshot if other
compatibility checks pass. Do not combine H1 and H2 in the initial tests.

H3 (earlier noise termination) and H4 (whole-feature masks) remain deferred.
Choose subsequent work after H1 evidence and remaining-budget accounting, not
as an unconditional four-way sweep.

## Validation and recovery

- Default loss values and parameter gradients match the pre-change formula
  exactly in the tested case. H1 preserves the MSE value and gradient exactly.
- H1 angular loss is independent of noise realization; H2 angular loss is
  invariant to cancelling member perturbations while individual MSE remains.
- Full suite: 42 passed; additional H1 runner integration test: 1 passed.
- Exact staged code/config/runtime hashes: `x34_h1_stage.json`.
- Kaggle computes `x34_review.json` itself after complete training; therefore
  result computation does not depend on a local monitor remaining alive.
- Partial training writes `x34_progress.json` and retains epoch checkpoints;
  no incomplete comparison or automatic relaunch.
- No submission in this stage. Champion and its CSV remain preserved.

Launch status is recorded separately in `x34_h1_status.json` after acceptance.

## Launch accepted

Kaggle accepted version 1 of `mscapital-x34-h1-tpu` with the 12,600-second
platform timeout. Initial API status: QUEUED. Source and stage provenance were
pushed before launch (commit `b936a6c`).

`watch_angular_screen.py --push --notify` can recover results and publish status
transitions. Desktop notification requires an awake, connected local computer;
it is not a guaranteed future chat message. Kaggle-side comparison still runs
without the watcher. Terminal failures or incomplete panels require inspection;
the watcher does not spend additional quota or submit predictions.

## H1 result — 2026-10-04: complete, not promoted

All eight models completed. Remote review recovered and recomputed locally with
checksum/row/recipe verification; deltas agree to floating-point precision.

| Pair | H1 minus control |
|---|---:|
| fold 0, seed 2026 | -0.000794 |
| fold 0, seed 42 | +0.001310 |
| fold 4, seed 2026 | +0.001616 |
| fold 4, seed 42 | +0.001766 |
| fold 0, two-seed mean | +0.000278 |
| fold 4, two-seed mean | +0.001574 |
| pooled rows | +0.002029 |

The pooled cosine is not an average of fold cosines. Its +0.002029 does not
override the fixed per-fold/seed gates, which FAIL. Month 66's delta is +0.007319;
do not infer broad transfer from the pooled number. No H1 confirmation or
submission. Machine-readable result: `x34_h1_review.json`.

Kaggle quota increased from 2.23 to 4.18 TPU hours used: about 1.95 h for H1
including overhead. Model timers sum to 1.797 h. No GPU training consumed.
Local watcher again did not deliver a verified completion notification; the
Kaggle-side comparison persisted correctly and was recovered in this session.

## H2 execution — next isolated hypothesis

Train only four H2 models (folds 0/4, seeds 2026/42), using the exact archived H1
trainer/modules byte-for-byte. Change only `angular_aggregation: members -> mean`.
Keep `angular_target: noisy`; do not combine H1 and H2.

Reuse H1's four complete controls after verifying their run signature, all
prediction checksums, dataset, source hashes, runtime and recipe. Controls are
attached read-only as a Kaggle kernel source; H1 predictions are not used as the
baseline. Reuse is justified because training code does not change at all.

Kernel: https://www.kaggle.com/code/diegoaranguren/mscapital-x34-h2-tpu
Soft stop 1.75 h; platform hard timeout 2 h. The cumulative maximum is then
4.18 + 2 = 6.18 TPU hours, within the previous 8.5 h authorization. GPU untouched.
The same fixed gates apply. Kaggle computes `x34_h2_review.json` after all four
new models complete. No next experiment or submission launches automatically.
Provenance: `x34_h2_stage.json`; status: `x34_h2_status.json`.
