# Next experiment: fixed recency weighting — 2026-10-03

Baseline: submitted flow31 no-clip, public LB 0.142, submission 56794864.

Hypothesis: models trained through more recent periods may transfer better to
the future scoring window than the oldest cutoffs. The five final train ends
are 37, 47, 52, 57 and 62; equal averaging gives each 20%. Keep every model and
every seed. Set fold weight proportional to `2 ** ((train_end - latest_end)/20)`.
The half-life is fixed before evaluating this candidate; do not tune it after
looking at results. Normalize weights to sum to one; equal weight within seeds.
Relative ages are identical at both confirmation origins and in the final.

Earlier aggregation experiments used different features and fold-specific OOF.
This is a bounded recheck on the audited flow31 ensemble, with all five models
scored on each external block. Prior failures remain evidence against a broad
search. This plan tests one candidate only, not greedy selection or optimization.

Evaluate complete recovered predictions for all three seeds, with checksums,
identical row sets, source/runtime/recipe compatibility between seed extensions,
and unchanged no-data handling. Both arms are unclipped. Report per-origin,
per-seed, monthly, LOMO and bootstrap uncertainty. These windows have already
been consulted; extra analyses do not create a fresh holdout.

Gate fixed in `configs/postprocessing/recency20.json`: delta >= 0.002 in each
origin's ensemble and each seed's pooled result; at least 6/9 months positive;
all LOMO deltas positive. If the gate fails, close this candidate, do not tune
the half-life, and do not submit it. If it passes, generate the exact candidate
CSV and report the evidence before a submission decision.

Budget: CPU only, zero additional GPU/TPU hours. The quota reset is not used to
expand the previously authorized accelerator budget. No remote job is launched.

## Executed result — closed, not promoted

Plan/config committed and pushed as `735ce14` before evaluating predictions.
Implementation: `src/kaggle/evaluate_recency.py`. Exact report:
`research/recency20_review.json`. Tests verify seed balancing, translation of
training cutoffs between origins, missing-model rejection and row alignment.

Fixed weights for folds 0–4: 0.118002, 0.166880, 0.198455, 0.236004, 0.280658.

| Comparison | Equal mean | Recency20 | Delta |
|---|---:|---:|---:|
| Pooled confirmation | 0.166282 | 0.167076 | +0.000795 |
| Origin 59 | — | — | +0.000923 |
| Origin 64 | — | — | +0.000545 |
| Seed 42, pooled | — | — | +0.000579 |
| Seed 123, pooled | — | — | +0.000894 |
| Seed 2026, pooled | — | — | +0.000807 |

Seven of nine monthly deltas are positive; minimum LOMO delta +0.000460.
Month-bootstrap interval [+0.000228, +0.001093], with the previously stated
dependence and selection limitations. This is a small consistent gain on the
consulted windows, not evidence of zero signal. It nevertheless fails the
prespecified +0.002 operational promotion threshold in both origins and seeds.

Decision: **do not promote, do not tune half-life, do not submit**. No candidate
CSV exported and no GPU/TPU time consumed. Champion remains the submitted
equal-weight no-clip ensemble, public LB 0.142. This experiment is complete;
no further training, monitor or submission was launched.

Reproduce:

```powershell
.venv/Scripts/python.exe src/kaggle/evaluate_recency.py --labels runs/recovered/labels --confirm runs/recovered/confirm/run runs/seed123/run --final runs/recovered/final/run --template runs/recovered/template/submission.csv --out runs/recency20
```
