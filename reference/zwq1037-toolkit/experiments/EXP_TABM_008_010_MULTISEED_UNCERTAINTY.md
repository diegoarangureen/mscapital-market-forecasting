# EXP-TABM-008--010: multi-seed and member-uncertainty checks

Date: 2026-09-11

## EXP-TABM-008: seed 137

This repeats EXP-TABM-001 with random seed 137 as the only intended model change.
The same 0--49 / 50--59 epoch-selection protocol selected 8 epochs, followed by
training on months 0--59 and evaluation on months 60--70.

| Model | Overall | 62--70 no66 | 67--70 | 60--64 | Primary std | Worst | q25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed 42 baseline | 0.176059 | 0.151066 | 0.149038 | 0.152685 | 0.014201 | 0.124334 | 0.145376 |
| Seed 137 | 0.167116 | 0.146990 | 0.144610 | 0.150801 | **0.012355** | **0.130385** | 0.137663 |

Seed 137 is smoother but materially weaker in the primary and recent slices.

## EXP-TABM-009: multi-seed blend

Centered seed-42 and seed-137 predictions have correlation 0.953916, so seed 137
does provide diversity. Alpha from 0.00 to 0.50 was tested in increments of 0.025,
both as a standalone TabM composite and inside the leading six-source maximin blend.

No nonzero alpha passed the robustness gates. Even alpha 0.025 reduced the six-source
overall score from 0.175986 to 0.175920, no66 from 0.154264 to 0.154235, recent from
0.153232 to 0.153204, and produced a negative primary LOMO minimum. Do not train a
full-data seed-137 model for submission.

## EXP-TABM-010: member-disagreement shrinkage

The 16 seed-42 TabM member predictions were used to estimate per-row uncertainty.
Twelve label-free transforms were checked: linear or exponential shrinkage by the
percentile rank of member standard deviation, plus mild hard shrinkage of the most
uncertain 5% or 10% of rows.

Member disagreement correlates 0.7014 with the absolute trimmed prediction. This
means high disagreement is often a consequence of large predicted signal rather
than pure unreliability. Every shrinkage variant lowered the six-source primary
score; zero shrinkage remained best and no candidate passed the robustness gates.

## Decision

- Reject seed-137 blending under the current protocol.
- Reject member-disagreement magnitude shrinkage.
- Keep `tabm_double_trim_b005_maximin_alpha005_fulltrain.csv` as the preferred
  prepared submission.

