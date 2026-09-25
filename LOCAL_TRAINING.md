# Local Training Handoff (Sep 25, 2026)

Everything needed to reproduce and extend the current model on a local machine.
Written for agents running on Diego's box while Kaggle TPU quota is exhausted
(resets weekly; a probe relaunches the queued kernels automatically).

## 1. Current model state (champion)

- **Architecture**: RealMLP (ensemblish MLP, N_ENS=16, PBLD periodic embeddings
  24 hidden / 3 out, trunk 3x NTPLinear d->512->512->128, GELU, dropout 0.01,
  diagonal feature mask). PyTorch, no torchvision extras.
- **Features**: 450f champion = 298 proprietary (full_train minus X21/X22 minus
  dropped cols) + 152 public (rfmf-0726data train.csv). Assembly order:
  `concat(full_train, X21, X22)`, then `np.delete([301, 267, 268, 299, 257], axis=1)`,
  then concat the 152 public cols. **X30 adds 36f -> 486f total.**
- **Scaler**: robust median/IQR, soft-clamped: `s = (X - med)/(IQR + 1e-30)`,
  `s / sqrt(1 + (s/3)^2)`; IQR==0 columns use half range, zero-var -> factor 0.
  Fit per fold on that fold's train split ONLY. (See `fit_scale`/`apply_scale`
  in `src/kaggle/realmlp_x30_pairrep_tpu.py`; exact code, do not paraphrase.)
- **Training recipe** (validated local optimum, keep fixed for comparable runs):
  AdamW grouped LRs (scale x20 wd 1e-3; num_embed x0.093 wd 1e-2; rest 1e-3 wd 1e-2;
  bias x0.1 wd 5e-3), betas (0.9, 0.98), BS 256, 10 epochs, flat-anneal LR
  (flat 50% then linear to 0), EMA 0.998, label noise 0.005 annealed per epoch,
  patience 3 on val cosine, grad clip 1.0.
  Loss: weighted MSE (w=0.5 where |y|>0.001) + 1.0 * (1 - cosine).
- **Validation protocol**: 5 purged folds, seed 2026:
  val months 40-44 / 50-54 / 55-59 / 60-64 / 65-70 with train <= 37/47/52/57/62.
  Metric: cosine similarity on val, and OOF cosine on months 61-70 and 66-70
  (66-70 is the window closest to test). Early stopping picks best-EMA epoch.
- **Reference per-fold baselines (seed 2026, 450f)**: f1 0.143251, f2 0.149152,
  f3 0.148980, f4 0.155356, f5 0.171549. OOF 61-70: 0.167176; 66-70: ~0.174.

## 2. Data sources (all Kaggle, all private to Diego's account except noted)

| What | Where | How to download locally |
|------|-------|------------------------|
| Base matrices (full_train.npy, full_y.npy, full_month.npy) | dataset `diegoaranguren/mscapital-matrices` | `kaggle datasets download -d diegoaranguren/mscapital-matrices` |
| X21 (36f) | `diegoaranguren/mscapital-x21` | same pattern |
| X22 (36f) | `diegoaranguren/mscapital-x22` | same pattern |
| X30 (36f + names) | `diegoaranguren/mscapital-x30` | same pattern |
| X31 (18f + names) | `diegoaranguren/mscapital-x31` | same pattern |
| Public 152f | kernel output `yunsuxiaozi/rfmf-0726data` (public) | `kaggle kernels output yunsuxiaozi/rfmf-0726data` (train.csv) |
| Raw streams (market/transaction/order feathers) | competition `ms-capital-real-financial-market-forecasting` | `kaggle competitions download -c ms-capital-real-financial-market-forecasting` |

Local auth needs Diego's own `~/.kaggle/kaggle.json` (Kaggle -> Settings ->
Create New Token). Never commit tokens; never paste them into chats or docs.
Competition download requires the account to have accepted the rules (it has).

## 3. Reproduce the pipeline locally

1. `pip install torch numpy pandas pyarrow` (torch 2.x, CPU or CUDA build).
2. Unpack datasets into one folder per dataset; note paths.
3. Assemble the 486f matrix exactly as in `src/kaggle/realmlp_x31_pairrep_tpu.py`
   lines "train features" (that script is the current reference implementation,
   including the FLOW-18 selection from X30 by feature name).
4. To build features from raw feathers instead of downloading the prebuilt
   datasets: `src/kaggle/build_x30.py` and `src/kaggle/build_x31.py`; run with
   `BASE=<raw dir> NS=1257637 SPLIT=train OUT=<out>` env vars. CPU-only, ~1-2h
   for X31-scale builds, single-threaded numpy is fine.
5. Train: the pairrep script is the reference. For local use set `SEEDS`,
   `FOLD_IDX` (e.g. `FOLD_IDX=0,1` to shard folds across agents), and point the
   `DATA`/`XTRA`/`XTRA2`/`XTRA3`/`XTRA4` env vars at local dirs. Remove/skip the
   XLA branches (`XLA=0`, device cuda/cpu). Everything else stays verbatim.

## 4. Local cost expectations (honest)

- Reference: one model (1 fold, seed 2026, 10 epochs, ~1.05M train rows, BS 256)
  takes **~17 min on Kaggle TPU v3-8**. The 10-model pairrep panel took 167 min;
  a 5-model ablation arm ~82 min; the staged X31 3-arm panel (15 models) ~4.2h.
- **Consumer GPU (RTX 3080/4080 class)**: expect roughly 1.5-3x slower than TPU
  per model at BS 256 (small batches underutilize the card). Full 15-model
  panel: ~7-13h. Shard by `FOLD_IDX` across agents/machines.
- **CPU only**: not recommended for the full panel (likely 8-20x slower; a
  single model can exceed 2-4h). Fine for feature builds and smoke tests.
- **Do NOT raise batch size to speed up**: the recipe is validated at BS 256.
  A bigger BS changes training dynamics and breaks comparability with the
  Kaggle baselines. If you must experiment with BS>=1024, treat it as a
  separate throughput experiment, never as the champion.
- The paired-panel logic still applies locally: run base and experimental arms
  in the SAME session/conditions so machine-level noise cancels. Verdict bar:
  consistent per-fold shift > 0.002 = signal, else flat.

## 5. Current evidence (what we know as of Sep 25 12:45)

- **X30 36f pack: SIGNAL CONFIRMED.** Paired same-session panel vs baselines:
  5/5 folds positive (f1 +0.0054, f2 +0.0013, f3 +0.0015, f4 +0.0022,
  f5 +0.0016; mean +0.0024). OOF 61-70 +0.0020, 66-70 +0.0020.
- **Family ablation**: FLOW 18f (OFI, order intensity, queue imbalance, depth
  ratios) carries ~60% and dominates extreme folds (f1 +0.0040, f5 +0.0029).
  VOL 8f (realized-vol estimators, spread) is FLAT (mean +0.0002, noise).
  PRICE 10f (trade-sign autocorr, micro/mid slopes, vwap/ticker) pending -
  blocked on TPU quota, auto-relaunches at reset.
- **Model class is closed**: GRU dead x3 (max 0.036 vs 0.10 bar), TabM flat,
  every public architecture scores below our 0.139 LB. Signal/features is the
  only lever. Cross-sectional features fail LB transfer 3x - absolute
  per-sample features only.
- Local feature gains repeatedly fail to transfer to LB: forward-sim gating is
  mandatory before any submission.

## 6. Prioritized experiment queue (for local agents)

1. **X31 screen (highest priority, TPU-blocked on Kaggle, runnable locally)**:
   3-arm paired panel base / FLOW / FLOW+X31 (`/tmp/kpush_x31s/script.py` =
   `src/kaggle/realmlp_x31_pairrep_tpu.py`). Question: does flow31 - flow show
   a consistent > 0.002 shift? ~4.2h TPU-equivalent, shardable by fold.
2. **PRICE ablation** (queued on Kaggle, auto-relaunches at TPU reset; do not
   duplicate locally unless weekend is long): same script with
   `X30_KEEP='price'` (see `src/kaggle/realmlp_x30_ablation_tpu.py`).
3. **Forward-sim gating of X30 486f** (design in research/X30_CONTINGENCY.md):
   multi-seed walk-forward on the confirmed pack; required before any
   submission ask. Needs the champion protocol verbatim.
4. **X32 candidates (research only)**: extend FLOW further along the X31
   directions that screen positive; 600s-window ticker aggregates; cross-sample
   structure WITHOUT cross-sectional normalization.

## 7. Rules that do not bend

- No submissions without Diego's explicit OK (gate: val >= 0.130 clear +
  robustness verified; no LB-noise-level experiments on the 5/day quota).
- Absolute per-sample features only (cross-sectional trap confirmed 3x).
- Competition data only; max team 1; never blend others' public prediction
  files; bestwater/ZWQ1037 datasets are private and off-limits.
- Commit + push every advance to the repo (main branch, Diego's identity).
