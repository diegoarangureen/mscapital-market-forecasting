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
