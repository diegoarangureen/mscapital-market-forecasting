# flow31 without prediction clipping — public LB 0.142

Created 2026-10-02; submitted and scored 2026-10-03.
New best public LB **0.142**, versus previous **0.141** (+0.001).
Submission ref **56794864**, status COMPLETE, verified via authenticated Kaggle
API. The submitted CSV matches the recorded candidate SHA256. The prior
submission 56634145 is retained. Private score is unknown.
Receipt: `submission_no_clip_2026-10-03.json`.

## Update — 2026-10-03: third-seed confirmation complete

All 10 additional TPU models completed. Their code, runtime, data, prediction
checksums and complete temporal coverage passed verification. Combined with
the original 20 flow31 confirmation models, the review now covers all three
seeds used by the final champion. Exact report: `seed123_review.json`.

| Three-seed confirmation | Original clipping | No clipping + no-data zeros | Delta |
|---|---:|---:|---:|
| All months 62–70 | 0.160361 | 0.166282 | +0.005920 |
| Origin 59, months 62–66 | 0.174104 | 0.180422 | +0.006318 |
| Origin 64, months 67–70 | 0.144534 | 0.149810 | +0.005276 |

Delta is positive in 6/9 months and all 9 leave-one-month-out comparisons.
Without month 66, delta remains +0.003489. Paired month-bootstrap 95% interval:
[+0.001000, +0.008331], subject to dependent months and prior selection.
The three-seed raw ensemble improves +0.000660 over the original two-seed raw
ensemble. These observations do not justify selecting the best individual seed
or retrospectively changing ensemble weights.

Recommendation: submit the already prepared **flow31_no_clip.csv** candidate for
one leaderboard test, preserving the 0.141 champion. This is a postprocessing
candidate; additional confirmation models do not change the final CSV. The
candidate SHA256 remains unchanged. Subsequently submitted with Diego's request
to proceed with submission versus further training; result recorded above.

Verified copy: `runs/candidate-no-clip-third-seed/flow31_no_clip.csv`.
The original `runs/candidate-no-clip/flow31_no_clip.csv` is identical.
Kaggle reports 2.23 TPU hours used in the refreshed quota; summed model timers
are 2.127 h excluding overhead. No GPU training was launched by this agent.
Quota refreshed to GPU 30 h / TPU 17.77 h available, but the existing user budget
is not automatically expanded. No further training has been launched.

The overnight local monitor stopped recording while the kernel was queued.
Completion was recovered and published manually in the next active session;
delivery of a Windows notification is not verified.

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
- Submission completed on 2026-10-03: public LB 0.142.

## Completed experiment design

Third-seed confirmation uses seed 123 already present in the final champion,
with the original archived source and TPU runtime. Ten models, all five folds
and both original origins, maximum three TPU hours. Existing two-seed controls
are reused only after exact code/runtime/data/recipe verification. This adds
seed robustness, not unseen temporal data or a new feature/loss candidate.

Kernel: https://www.kaggle.com/code/diegoaranguren/mscapital-confirm-seed123

`seed123_status.json` records COMPLETE / review_complete, and
`seed123_review.json` contains the verified combined review. No replacement
kernel was launched. A single submission was subsequently sent and scored.

## Reproduce

```powershell
.venv/Scripts/python.exe src/kaggle/review_champion.py --labels runs/recovered/labels --confirm runs/recovered/confirm/run --final runs/recovered/final/run --template runs/recovered/template/submission.csv --out runs/candidate-no-clip
```

After completion, add `runs/seed123/run` after the original `--confirm` directory
and use a separate output directory. No training matrices or secrets are
committed; the scripts, provenance, checksums and evaluation reports are tracked.
