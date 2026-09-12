# MSCapital: Real Financial Market Forecasting

Kaggle competition solution and write-up for
[MS Capital: Real Financial Market Forecasting](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting).

Public leaderboard: **0.110** (rank 197/206 as of Sep 11, 2026). Validation cosine: **0.1274**.

## The problem

Each sample is a 10-minute window of high-frequency activity for one asset:
raw trades (price, size, aggressor side), order-book events (L1/L2), and
market bars. The target is the forward return of the window, and the
competition metric is **cosine similarity** between the predicted and true
target vectors, not a per-sample error. The task is therefore to rank a
population of ~650k test windows so that the prediction *vector* points in
the same direction as the realized-return vector - global direction matters,
per-sample magnitude much less.

## Approach

- **Honest validation.** Train months 0-60, validate on months 61-70, and a
  robustness check on a shifted split (train 0-50, validate 51-60, cosine
  0.1208) to confirm the validation estimate is not a lucky month range.
- **58 microstructure features per sample**, built in a single constant-memory
  streaming pass: trade/order/book aggregates, multi-horizon bar returns
  (30s to 10min), volume windows and acceleration, trend slopes, volatility,
  spread and book-imbalance means, short-horizon trade-flow imbalance, price
  dispersion, recency-weighted trade timing.
- **LightGBM** with bagging 0.7 (chosen by coordinate sweep), early stopping
  on validation, 3-seed averaging, refit on all labelled months with 1.1x
  iterations.
- **Cosine-aware post-processing:** mean-center predictions (the target is
  approximately mean-zero, so centering adds ~0.0016 for free), zero the 17
  test samples with no underlying data, clip to the 0.1/99.9 percentile band
  of train predictions.

## Memory constraint as an engineering problem

The data ships as single-batch LZ4-compressed feather files whose decompressed
record batches exceed 2 GB of RAM, so `pyarrow` cannot materialize them. The
repo therefore includes a minimal flatbuffers/Arrow-IPC parser
(`src/arrow_parse.py`) and chunked column readers (`src/stream_columns.py`)
that decompress one buffer chunk at a time, letting the feature builder run
in constant memory.

## Reproduce

```bash
bash src/download_data.sh          # needs a Kaggle Bearer token in ~/.kaggle/access_token
pip install -r requirements.txt
python3 src/build_features.py train 1257637 /path/to/data /path/to/work
python3 src/build_features.py test  647896  /path/to/data /path/to/work
python3 src/train_model.py /path/to/data /path/to/work   # writes submission.csv
```

## What did NOT work (and why it matters)

A 30-experiment ledger with numbers is in [EXPERIMENTS.md](EXPERIMENTS.md).
The short version: L2 order-book features, additional microstructure signals
(11 features), trade-sign inference (Lee-Ready tick rule, bulk volume
classification), extra-trees, and dart boosting all landed at or below the
champion; hyper-parameter differences under ~0.002 validation cosine are
seed noise. The provided `side` column already is the true aggressor
(IC +0.075 vs +0.027 for the best tick-rule reconstruction), and no two
samples share the same underlying market window, so there is no cross-sample
leakage to harvest legitimately. The remaining gap to the top of the
leaderboard (0.172) looks structural - cross-asset factors or sequential
models - not a tuning problem.
