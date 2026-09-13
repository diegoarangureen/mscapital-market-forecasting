# one pass over market.feather collecting per-sample coarse fingerprints (returns + log-volume)
# RAM-light: float32, NG grid points; processes both splits
import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter

NG = 30
grid = -np.arange(590, 0, -20.0)[:NG]   # descending sbp -> ascending -sbp

def collect(split, ns):
    cnt = np.zeros(ns, np.int64)
    first_sbp = np.full(ns, np.nan, np.float32)
    t0 = time.time()
    # per-sample running buffers (dict of lists is too slow; use full-length arrays)
    # market rows sorted by sid: process sample blocks; collect raw bars into per-sample
    # arrays via np.add.at style accumulation on the fly is messy -> store bars sparsely:
    # instead: accumulate per-sample (sum, count) per grid cell by assigning each bar to nearest grid cell
    cntg = np.zeros((ns, NG), np.int32)
    sump = np.zeros((ns, NG), np.float32)
    sumv = np.zeros((ns, NG), np.float32)
    for sid, sbp, ap, tv in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume'], elems=1<<19):
        g = np.searchsorted(grid, -sbp.astype(np.float64))
        np.clip(g, 0, NG-1, out=g)
        np.add.at(sump, (sid, g), ap.astype(np.float32))
        np.add.at(sumv, (sid, g), tv.astype(np.float32))
        np.add.at(cntg, (sid, g), 1)
        np.add.at(cnt, sid, 1)
    R = np.where(cntg>0, sump/np.maximum(cntg,1), np.nan).astype(np.float32)
    V = np.where(cntg>0, sumv/np.maximum(cntg,1), 0).astype(np.float32)
    print(split, 'collect', round(time.time()-t0), 's', flush=True)
    np.save(f'/tmp/work/fpR_{split}.npy', R)
    np.save(f'/tmp/work/fpV_{split}.npy', V)
    np.save(f'/tmp/work/fpcnt_{split}.npy', cnt)
    del R, V, sump, sumv, cntg; gc.collect()

if __name__ == '__main__':
    collect('train', 1257637)
    collect('test', 647896)
    print('FP_DONE', flush=True)