# Forward-sim gating protocol (pre-submission, X30 486f candidate)
Written 2026-09-25 13:00, post pairrep signal + FLOW/VOL ablation.

Purpose: the paired k-fold panel measures signal; forward-sim measures whether
that signal SURVIVES the actual submission conditions (retrain-on-full style,
recent-window generalization, seed luck). No submission ask without it.
Submission gate (parent): val >= 0.130 clear + robustness verified; no
LB-noise-level experiments on the 5/day quota.

## Candidate
Champion recipe verbatim (LOCAL_TRAINING.md section 1), 486f = 450 + X30 36f.
If the X31 screen lands first with signal, the candidate becomes 504f and this
protocol re-runs on it instead (same design).

## Protocol A — multi-seed paired stability (8.5h TPU-equiv)
5 folds x 3 seeds (2026, 7, 99), arms base vs 486f in the SAME session
(2 arms x 5 folds x 3 seeds = 30 models). Shard by FOLD_IDX across agents.
PASS: per-seed mean delta (486f - base) > +0.002 for every seed, and
5/5 folds positive in at least 2 of 3 seeds.
FAIL (any seed with negative mean): signal is partly seed luck -> shrink
candidate (e.g. FLOW-only 18f) and re-run A on the subset before asking.

## Protocol B — latest-window trajectory (1.7h TPU-equiv)
Single purged split: train months <= 57, validate 58-70 (post-champion
territory, the most test-like data we have), 3 seeds, arms base vs 486f same
session (6 models). Also emit the per-month OOF delta curve on months 61-70
from Protocol A (486f - base per month): PASS = B delta > +0.002 mean over
seeds AND no systematic decay in the per-month curve toward month 70 (signal
must not be dying as we approach test). A decaying curve = expect LB transfer
weaker than k-fold suggests; report, do not submit.

## Expected LB mapping (for the submission ask)
Champion: OOF 61-70 0.167 -> LB 0.139 (val-to-LB haircut ~0.028).
486f forward-sim OOF target: >= 0.169 (i.e. +0.002 over champion in BOTH
protocols) -> projected LB ~0.141. Report projected LB with the haircut
stated; Diego/parent decide if the submission slot is worth it (5/day,
resets ~02:00 CEST; competition deadline Oct 9 16:00 UTC).

## Cost plan (weekly TPU 20h, post-reset queue)
1. PRICE ablation 1.4h (auto via quota probe)
2. X31 3-arm screen 4.2h (auto-chained; Diego's local agents may land it first)
3. Forward-sim A+B ~10.2h (this doc) -> then submission ask if gates pass
Total ~15.8h, fits with ~4h spare. If X31 screens positive, candidate changes
and forward-sim runs on 504f instead (same cost).
