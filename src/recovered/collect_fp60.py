# 60-grid fingerprints (10s spacing), two passes per split to stay in RAM.
import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter
NG = 60
grid = -np.arange(590, 0, -10.0)[:NG]
def collect(split, ns):
    cntg = np.zeros((ns, NG), np.int32)
    sump = np.zeros((ns, NG), np.float32)
    cnt = np.zeros(ns, np.int64)
    t0 = time.time()
    for sid, sbp, ap in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','transaction_avgprice'], elems=1<<19):
        g = np.searchsorted(grid, -sbp.astype(np.float64))
        np.clip(g, 0, NG-1, out=g)
        np.add.at(sump, (sid, g), ap.astype(np.float32))
        np.add.at(cntg, (sid, g), 1)
        np.add.at(cnt, sid, 1)
    R = np.where(cntg>0, sump/np.maximum(cntg,1), np.nan).astype(np.float32)
    del sump; gc.collect()
    np.save(f'/tmp/work/fp60R_{split}.npy', R)
    print(split, 'pass1 price', round(time.time()-t0), 's', flush=True)
    sumv = np.zeros((ns, NG), np.float32)
    for sid, sbp, tv in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','transaction_volume'], elems=1<<19):
        g = np.searchsorted(grid, -sbp.astype(np.float64))
        np.clip(g, 0, NG-1, out=g)
        np.add.at(sumv, (sid, g), tv.astype(np.float32))
    V = np.where(cntg>0, sumv/np.maximum(cntg,1), 0).astype(np.float32)
    del sumv, cntg; gc.collect()
    np.save(f'/tmp/work/fp60V_{split}.npy', V)
    print(split, 'pass2 volume', round(time.time()-t0), 's', flush=True)
    del R, V, cnt; gc.collect()
if __name__ == '__main__':
    collect('train', 1257637)
    collect('test', 647896)
    print('FP60_DONE', flush=True)
