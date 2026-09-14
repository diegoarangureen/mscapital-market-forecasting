# 60-grid fingerprints (10s spacing), sample-blocked bincount, in-place finalize.
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NG = 60
grid = -np.arange(590, 0, -10.0)[:NG]
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split('kB')[0])//1024
def collect(split, ns):
    t0 = time.time()
    sump = np.zeros((ns, NG), np.float32)
    cntg = np.zeros((ns, NG), np.int32)
    for k, (sid, sbp, ap) in enumerate(ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','transaction_avgprice'], elems=1<<20)):
        g = np.searchsorted(grid, -sbp.astype(np.float64)); np.clip(g, 0, NG-1, out=g)
        s0 = int(sid.min()); s1 = int(sid.max())
        loc = (sid.astype(np.int64)-s0)*NG + g
        nb = (s1-s0+1)*NG
        sump[s0:s1+1] += np.bincount(loc, weights=ap.astype(np.float64), minlength=nb).reshape(-1, NG).astype(np.float32)
        cntg[s0:s1+1] += np.bincount(loc, minlength=nb).reshape(-1, NG).astype(np.int32)
        if k % 50 == 0: _libc.malloc_trim(0)
    print(split, 'p1 loop done', round(time.time()-t0), 'rss', rss(), flush=True)
    for st in range(0, ns, 200000):
        en = min(st+200000, ns)
        c = cntg[st:en]
        np.divide(sump[st:en], np.maximum(c,1), out=sump[st:en])
        sump[st:en][c==0] = np.nan
    np.save(f'/tmp/work/fp60R_{split}.npy', sump)
    del sump; gc.collect(); _libc.malloc_trim(0)
    print(split, 'pass1 price saved', round(time.time()-t0), 'rss', rss(), flush=True)
    sumv = np.zeros((ns, NG), np.float32)
    for k, (sid, sbp, tv) in enumerate(ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','transaction_volume'], elems=1<<20)):
        g = np.searchsorted(grid, -sbp.astype(np.float64)); np.clip(g, 0, NG-1, out=g)
        s0 = int(sid.min()); s1 = int(sid.max())
        loc = (sid.astype(np.int64)-s0)*NG + g
        nb = (s1-s0+1)*NG
        sumv[s0:s1+1] += np.bincount(loc, weights=tv.astype(np.float64), minlength=nb).reshape(-1, NG).astype(np.float32)
        if k % 50 == 0: _libc.malloc_trim(0)
    print(split, 'p2 loop done', round(time.time()-t0), 'rss', rss(), flush=True)
    for st in range(0, ns, 200000):
        en = min(st+200000, ns)
        c = cntg[st:en]
        np.divide(sumv[st:en], np.maximum(c,1), out=sumv[st:en])
        sumv[st:en][c==0] = 0.0
    np.save(f'/tmp/work/fp60V_{split}.npy', sumv)
    del sumv, cntg; gc.collect(); _libc.malloc_trim(0)
    print(split, 'pass2 volume saved', round(time.time()-t0), 'rss', rss(), flush=True)
if __name__ == '__main__':
    collect('train', 1257637)
    gc.collect(); _libc.malloc_trim(0)
    collect('test', 647896)
    print('FP60_DONE', flush=True)
