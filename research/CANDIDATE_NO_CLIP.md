# flow31 without prediction clipping — candidate, not submitted

Date: 2026-10-02. Champion remains public LB **0.141**.

Recovered all 15 final and 40 original confirmation prediction artifacts from
the owner's completed Kaggle kernels. Verified prepared label/ID hashes,
prediction checksums, job manifests, temporal masks and ensemble coverage.

## Reproduced validation

| Previously consulted confirmation data | Champion clipping | No clipping + no-data zeros | Delta |
|---|---:|---:|---:|
| All months 62–70 | 0.160042 | 0.165621 | +0.005579 |
| Origin 59, months 62–66 | 0.173953 | 0.179760 | +0.005808 |
| Origin 64, months 67–70 | 0.143988 | 0.149105 | +0.005117 |

Exact machine-readable results: `champion_no_clip_review.json`.
Paired month-bootstrap 95% interval for the overall delta:
[+0.000908, +0.007814]. Few dependent months and prior selection limit this
interval. It is not evidence of a specific leaderboard gain. These are
checkpoint-external windows but are not a new holdout.

## Export

- Reconstructed submitted champion hash:
  `c2e5e71c03bcc75ce0d79f3edcbae764228e16553b30778927a3091a431e503a`.
  Matches the historical recorded 16-character prefix.
- Candidate: `runs/candidate-no-clip/flow31_no_clip.csv` (ignored local artifact).
- Candidate SHA256:
  `7c59c37ec4de110ac368f2bc6bde4d31f1202b97480720081e52de2a1b7611ee`.
- Official template order and columns; 647,896 rows, all finite.
- Same 15 models and arithmetic mean; only prediction clipping removed.
- 742 rows differ from the champion; 17 no-data rows remain zero.
- No submission has been made.

## In progress

Third-seed confirmation uses seed 123 already present in the final champion,
with the original archived source and TPU runtime. Ten models, all five folds
and both original origins, maximum three TPU hours. Existing two-seed controls
are reused only after exact code/runtime/data/recipe verification. This adds
seed robustness, not unseen temporal data or a new feature/loss candidate.

Kernel: https://www.kaggle.com/code/diegoaranguren/mscapital-confirm-seed123

`seed123_status.json` records remote status; `seed123_review.json` will be written
only after all ten additional models complete and artifact verification passes.
The local watcher can publish these reports and request a Windows notification.
It does not submit, launch a replacement kernel or spend additional quota.

## Reproduce

```powershell
.venv/Scripts/python.exe src/kaggle/review_champion.py --labels runs/recovered/labels --confirm runs/recovered/confirm/run --final runs/recovered/final/run --template runs/recovered/template/submission.csv --out runs/candidate-no-clip
```

After completion, add `runs/seed123/run` after the original `--confirm` directory
and use a separate output directory. No training matrices or secrets are
committed; the scripts, provenance, checksums and evaluation reports are tracked.
