# MSCapital: Real Financial Market Forecasting

**Audited training pipeline (Sep 25):** follow [TRAINING_AGENT.md](TRAINING_AGENT.md)
for the corrected builders, shared CPU/CUDA/TPU runner, temporal confirmation,
checkpoint resume, tests and training budget. Historical scores below were
produced by the earlier pipeline; they are not rescored audit results. In
particular, the old RealMLP `cos_np` was Pearson and aggregated OOF retained the
last seed. Active X30/X31 builders now emit **X30v2/X31v2**; rebuild both splits.

End-to-end research project for the Kaggle competition
[MS Capital: Real Financial Market Forecasting](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
(~650k high-frequency market windows; metric: cosine similarity between the
predicted and realized return vectors).

**Current standing: public leaderboard 0.139, rank 131/244** (as of Sep 24, 2026).
Top of board is 0.180; top-10 cut is 0.164. Work is active and updated daily.

## Results trajectory

| Date | Submission | Recipe | Public LB |
|------|-----------|--------|-----------|
| Sep 11 | v6 | LightGBM, 58 microstructure features | 0.110 |
| Sep 15 | v9 | RealMLP 246f, refit-on-full | 0.124 |
| Sep 17 | v13 | RealMLP 450f, 5 purged folds x 3 seeds, holdout + early stopping, 15-model average | **0.139** |
| Sep 18 | v14 | v13 + 75 cross-sectional rank features (XS75) | 0.135 (negative; see below) |
| Sep 20 | v15 | v13 recipe, 5 folds x 5 seeds (25 models) | 0.138 (flat; OOF +0.0005 did not transfer) |

Since v14, eight controlled validation A/Bs on the champion (order-flow stream
aggregates, masked microstructure + liquidity-void features, loss reweighting,
model capacity, correlation pruning, window weighting, clipping, seed
replication) all landed within the fold-to-fold noise floor (~0.004 cosine) -
documented with numbers in the log. Seed replication confirmed a +0.002
"improvement" was pure seed noise, which closed the incremental
feature-stacking line. Current work: a 25-model k-fold OOF run (5 folds x 5
seeds) for per-row uncertainty estimation and offline prediction calibration
(shrinkage). v15 tested the aggregation ceiling directly: 25 models instead
of 15 moved OOF +0.0005 and the leaderboard not at all (0.138 vs 0.139) -
with a 25-model OOF tensor available, every post-hoc calibration tried
(shrinkage, robust aggregation, model selection) was neutral or harmful.
The equal-weight k-fold mean is the ceiling for this recipe; further gains
have to come from new features or architecture.

Latest research (Sep 24): the controlled screen line closed three
independent feature/preprocess hypotheses - RQ-KMeans auxiliary targets
(-0.0003), empirical-quantile feature scaling (mixed, within noise), and
price=0 empty-book masking (4 of 5 folds within +-0.0002 of baseline) - all
on a 5-fold single-session TPU panel validated against per-fold baselines
(the panel reproduces reference folds to +-0.0002). A full scan of the
public discussion landscape (research/R1_METHODS_LANDSCAPE.md) re-framed
the problem: the board leader states he uses competition data only, so the
~0.16 "public plateau" is a method ceiling, not a data ceiling. Cross-sectional
feature traps were independently confirmed a third time (CV 0.151 -> LB 0.136).
X30, an Optiver-canon proprietary order-flow feature pack
(multi-level order-flow imbalance, trade-sign autocorrelation, realized-vol
estimators, order arrival/cancel intensity, queue-imbalance dynamics), was
built from the raw streams and screened with the same forward protocol.

Latest (Sep 25): X30 is the first proprietary feature family to clear the
paired-panel gate - all 5 folds positive, mean +0.0024 cosine over the
450-feature champion baseline, with per-fold baselines reproducing to six
decimals in-session. A family ablation then attributed the gain: the
order-flow group (OFI, order intensity, queue imbalance, depth ratios)
carries about 60% of it and dominates the extreme folds, the volatility
group is flat (mean +0.0002, inside session noise), and the price-
microstructure group is queued. Current line: X31, an 18-feature extension
of the order-flow family (sub-bucket flow burstiness and acceleration,
time-weighted OFI, volume-weighted order flow, inter-arrival gap
statistics, Kyle lambda, VPIN-lite), screened as a 3-arm paired panel
(base / flow / flow+extension) so the incremental question is measured
free of session noise.
Historical note: the "455f" tag was an early miscount; the build is 450
columns (298 proprietary + 152 public).

Every experiment - including the dead ends - is logged with numbers in
[research/LOG.md](research/LOG.md) and [EXPERIMENTS.md](EXPERIMENTS.md).

## The problem

Each sample is a 10-minute window of high-frequency activity for one asset:
raw trades (price, size, aggressor side), order-book events (L1/L2), and
market bars. The target is the forward return of the window, and the metric
is **cosine similarity** between prediction and target vectors, not a
per-sample error - so the task is to make the prediction *vector* point in
the same direction as the realized-return vector across ~650k test windows.

## Current approach (v13, LB 0.139)

- **Features (450).** 298 proprietary microstructure features (trade, order
  flow, book, multi-horizon bar statistics) with adversarial-validation
  pruning of regime-shifted columns, plus 152 public domain features shared
  by the community (yunsuxiaozi's rfmf-0726 set). Sample-id alignment to the
  competition matrices is verified by permutation assertions in-kernel.
- **Model.** RealMLP - an ensemble-MLP with periodic
  boundary/linear-block embeddings (PBLD), NTP-linear layers and EMA
  weights - trained with a weighted MSE + cosine hybrid loss on the raw
  target, label-noise regularization, grouped learning rates and a
  flat-then-anneal schedule.
- **Inference protocol.** 5 purged time-series folds (validation months
  40-44/50-54/55-59/60-64/65-70, training restricted to earlier months)
  x 3 seeds, every model holdout + early-stopped on its own validation
  window; the submission is the average of all 15 models. This is the
  protocol that finally transferred validation gains (+0.018 cosine) to the
  leaderboard (+0.015) after four procedurally different submissions had
  landed flat at 0.123-0.124.
- **Post-processing.** Zero the 17 no-data test rows, clip to the 0.1/99.9
  percentile band of train predictions.

## What the experiment ledger shows

Closed research lines, each with numbers in the log: cross-sectional /
transductive feature families (within-month ranks computed against global
test - they lift validation and OOF cosine but lose on the public
leaderboard, v13 0.139 vs v14 0.135), sequence models (GRUs
over uniform 1s grids, incl. order-flow streams), adversarial feature
pruning beyond the accepted 5 columns, DANN gradient-reversal domain
adaptation, online learning (impossible in this submission format),
auxiliary targets from future bars, multi-head MLP (TabM-style)
architectures, and ensembles that dilute the champion. The distribution
shift between late-train and test is diffuse (adversarial AUC stays ~0.78
after pruning), which is why alignment approaches fail and feature
engineering is where the gains live.

## Engineering notes

- The data ships as single-batch LZ4-compressed feather files whose
  decompressed record batches exceed 2 GB, so the repo includes a minimal
  flatbuffers/Arrow-IPC parser (`src/arrow_parse.py`) and chunked column
  readers for constant-memory feature building.
- The full research loop (feature builds, GPU training kernels, validation,
  Kaggle submission, leaderboard tracking) is automated end-to-end; kernels
  and datasets are versioned in `src/kaggle/`.
- Long GPU runs checkpoint incrementally (per-model OOF/test prediction
  stacks) after a 10-hour session-timeout loss taught the pipeline to never
  save only at the end.

## Repo map

- `src/` - feature builders, trainers, the Arrow-IPC streaming parser
- `src/kaggle/` - Kaggle kernels (validation and submission) and API tooling
- `research/` - experiment log, discussion mining, reference reimplementations
- `EXPERIMENTS.md` - the 30+ experiment ledger with numbers

## Reproduce

The commands below reproduce the historical baseline. For current training,
use the explicit configurations and commands in **TRAINING_AGENT.md**.

```bash
bash src/download_data.sh          # needs a Kaggle Bearer token in ~/.kaggle/access_token
pip install -r requirements.txt
python3 src/build_features.py train 1257637 /path/to/data /path/to/work
python3 src/build_features.py test  647896  /path/to/data /path/to/work
python3 src/train_model.py /path/to/data /path/to/work   # writes submission.csv
```
