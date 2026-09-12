# MSCapital state (rebuilt 2026-09-12 ~16:30 CEST after sandbox wipe)

## Champion (unchanged)
GBM v4: X2 58 feats, LGBM bag0.7, 3-seed avg, refit 0-70 @1.1x iters, mean-center, 17 nodata zeroed, clip 0.1/99.9 -> val 0.127385, LB 0.110, rank 197/206. Submission bar: val >= 0.130 clear.

## Data/scripts status
- /tmp/mscapital: full re-download done (8 files).
- /tmp/work scripts: arrowparse (slot-1 fix), streamcol, streamcol2, features2 (13-col fix), features6 (OFI/microprice/Kyle - UNTESTED), collect_fp (fingerprints), diag_slots, train_gbm, combine, submit.
- GitHub backup: https://github.com/diegoarangureen/mscapital-market-forecasting (PAT in vault "GitHub PAT - mscapital backup").
- Sandbox has only ~2GB RAM: use combine.py (memmap), never part_mk.py; one heavy job at a time.

## Running
- /tmp/rebuild3.sh (pid was 2649) log /tmp/x3.log: combine train/test -> features6 train/test -> collect_fp -> PIPELINE_ALL_DONE.
- Recurring wake wakeschedule-01M2AZC89MGT2AMXP70KRCN9NN every 45min.

## STRUCTURAL FINDING (16:20): cross-asset time slots exist
Month-61: every sample has cross-sample twins with bar-return corr>0.4 (median twin 0.47). Assets share wall-clock windows -> market factor recoverable from raw data (no labels) in train AND test.
NEXT: cluster samples into slots (chunked corr, thr ~0.55, union-find) -> slot variance share of y -> slot-context features -> X7 GBM.

## Diagnostics so far
- Month y-means tiny but month70 mean = 16 SE from 0 (market drift exists within months).
- y autocorr ~0 at all lags within month (sample_id not time-ordered).
- Prices normalized ~1.0 (4658 month-61 samples have px=0 gaps = no-data samples?); 179 bars/10-min window, event-driven.

## SLOT FINDING RESOLVED - NEGATIVE (16:47)
Twin pairs by fingerprint corr (0.3-0.5) show label corr 0.005-0.007 vs random 0.008: forward returns do NOT co-move across samples with similar windows. Fingerprint twins are selection noise (max over 3k candidates, noise ceiling ~0.36 at 117 dims). No market factor in the TARGET. Slot/market-factor path closed. Lesson: validate with labels before building.
Remaining levers: (1) X6 OFI/microprice/Kyle per-sample features (in pipeline), (2) sequence models on raw 179-bar series (CPU-only: days - likely what leader does), (3) minor: calibration/blends.

## 16:55 update
- X2 rebuilt (train+test, 58 feats, keys identical to champion). X6 built: kyle_lambda, ofi1, ofi1_late, ofi2 (CKS event-OFI from market bars L1/L2, Kyle lambda from signed trades).
- Univariate ICs: kyle -0.0004, ofi1 +0.0043, ofi1_late +0.0111, ofi2 -0.0104 (weak; GBM combo pending).
- eval6.py running: X2+X6 champion-params 3-seed -> compare vs 0.125774/0.127385.
- Fingerprints fpR/fpV collected for train+test (30-pt grid) - reusable for any future slot/cluster work (but slot path closed on label evidence).

## 17:06 - X6 NEGATIVE
X2+X6 seedavg centered val 0.118248 (seeds 0.1137/0.1124/0.1152) vs champion 0.127385. OFI/Kyle features hurt. CKS/microprice/Kyle path closed at GBM level.
NEXT: verify X2 rebuild reproduces champion (single seed should be ~0.1232); then next structural lever.
