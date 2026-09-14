# X11: order-free microstructure. 4 sbp buckets (60/180/300): hi-lo price range/mean,
# traded volume range (max-min cumtv), sbp span; + global min sbp (time-to-event) + total span. 14 feats.
import sys, numpy as np, time, ctypes, gc
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
def build(split, ns):
    t0 = time.time()
    N = ns*NB
    f64 = np.float64
    n = np.zeros(N, np.int64)
    maxp = np.full(N, -1e18, f64); minp = np.full(N, 1e18, f64)
    maxtv = np.full(N, -1e18, f64); mintv = np.full(N, 1e18, f64)
    maxt = np.full(N, -1e18, f64); mint = np.full(N, 1e18, f64)
    sump = np.zeros(N, f64)
    gmin = np.full(ns, 1e18, f64); gmax = np.full(ns, -1e18, f64)
    cols = ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume']
    for ch in ziter(f'/tmp/mscapital/{split}/market.feather', cols, elems=1<<19):
        (sid, sbp, ap, tv) = ch
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        t = sbp.astype(f64); x = ap.astype(f64); v = tv.astype(f64)
        n += np.bincount(lin, minlength=N)
        sump += np.bincount(lin, weights=x, minlength=N)
        np.maximum.at(maxp, lin, x); np.minimum.at(minp, lin, x)
        np.maximum.at(maxtv, lin, v); np.minimum.at(mintv, lin, v)
        np.maximum.at(maxt, lin, t); np.minimum.at(mint, lin, t)
        np.minimum.at(gmin, sid.astype(np.int64), t)
        np.maximum.at(gmax, sid.astype(np.int64), t)
        _libc.malloc_trim(0)
    print(split, 'stream done', round(time.time()-t0), 's', flush=True)
    nn = np.maximum(n, 1).astype(f64)
    meanp = sump/nn
    empty = n==0
    feats = {}
    for bi in range(NB):
        sl = slice(bi, None, NB)
        rng = (maxp[sl]-minp[sl])/np.maximum(meanp[sl],1e-9)
        vol = np.maximum(maxtv[sl]-mintv[sl], 0.0)
        span = np.maximum(maxt[sl]-mint[sl], 0.0)
        e = empty[sl]
        for k, vv in ((f'r{bi}_hlrange',rng),(f'r{bi}_tvrange',vol),(f'r{bi}_sbpspan',span)):
            vv = vv.astype(np.float32); vv[e] = 0.0; feats[k] = vv
    gm = gmin.copy(); gm[~np.isfinite(gm)] = 0.0
    gs = gmax-gmin; gs[~np.isfinite(gs)] = 0.0
    feats['r_gap_to_predict'] = gm.astype(np.float32)
    feats['r_total_span'] = gs.astype(np.float32)
    names = sorted(feats.keys())
    X = np.empty((ns, len(names)), np.float32)
    for j,k in enumerate(names): X[:,j] = feats[k]
    np.save(f'/tmp/work/X11_{split}.npy', X)
    np.save(f'/tmp/work/X11_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
