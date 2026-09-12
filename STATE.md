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
