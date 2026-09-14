# X10: finer sbp bucketization (6 buckets) + per-bucket first/last price return.
# Buckets: [0,30),[30,90),[90,180),[180,300),[300,450),[450,600). 10 stats/bucket -> 60 feats.
import sys, numpy as np, time, ctypes, gc
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 6
edges = np.array([30.0, 90.0, 180.0, 300.0, 450.0])
def bucket(sbp):
    return np.searchsorted(edges, sbp)
def build(split, ns):
    t0 = time.time()
    N = ns*NB
    f64 = np.float64
    n   = np.zeros(N, np.int64)
    st=np.zeros(N,f64); st2=np.zeros(N,f64); sp=np.zeros(N,f64); sp2=np.zeros(N,f64); stp=np.zeros(N,f64)
    sspread=np.zeros(N,f64); srel=np.zeros(N,f64); simb1=np.zeros(N,f64); simb2=np.zeros(N,f64); smicro=np.zeros(N,f64)
    maxtv=np.zeros(N,f64)
    maxsbp=np.full(N,-1.0,f64); minsbp=np.full(N,1e18,f64)
    firstp=np.zeros(N,f64); lastp=np.zeros(N,f64)
    cols = ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume',
            'ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
            'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2']
    for ch in ziter(f'/tmp/mscapital/{split}/market.feather', cols, elems=1<<19):
        (sid, sbp, ap, tv, a1, b1, av1, bv1, a2, b2, av2, bv2) = ch
        b = bucket(sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        t = sbp.astype(f64); x = ap.astype(f64)
        a1f = a1.astype(f64); b1f = b1.astype(f64)
        mid = (a1f+b1f)/2
        n   += np.bincount(lin, minlength=N)
        st  += np.bincount(lin, weights=t, minlength=N)
        st2 += np.bincount(lin, weights=t*t, minlength=N)
        sp  += np.bincount(lin, weights=x, minlength=N)
        sp2 += np.bincount(lin, weights=x*x, minlength=N)
        stp += np.bincount(lin, weights=t*x, minlength=N)
        sspread += np.bincount(lin, weights=(a1f-b1f), minlength=N)
        srel += np.bincount(lin, weights=(a1f-b1f)/np.maximum(mid,1e-9), minlength=N)
        imb1 = (bv1.astype(f64)-av1)/(bv1.astype(f64)+av1+1)
        bvv = bv1.astype(f64)+bv2; avv = av1.astype(f64)+av2
        imb2 = (bvv-avv)/(bvv+avv+1)
        simb1 += np.bincount(lin, weights=imb1, minlength=N)
        simb2 += np.bincount(lin, weights=imb2, minlength=N)
        denom = np.maximum(av1.astype(f64)+bv1, 1)
        micro = (a1f*bv1 + b1f*av1)/denom
        smicro += np.bincount(lin, weights=micro/np.maximum(mid,1e-9)-1, minlength=N)
        np.maximum.at(maxtv, lin, tv.astype(f64))
        # first (largest sbp in bucket) / last (smallest sbp) price tracking
        upd_first = t > maxsbp[lin]
        idx = lin[upd_first]
        maxsbp[idx] = t[upd_first]; firstp[idx] = x[upd_first]
        upd_last = t < minsbp[lin]
        idx = lin[upd_last]
        minsbp[idx] = t[upd_last]; lastp[idx] = x[upd_last]
        _libc.malloc_trim(0)
    print(split, 'stream done', round(time.time()-t0), 's', flush=True)
    nn = np.maximum(n, 1).astype(f64)
    B = 1<<20
    # in-place reuse: slope->stp, pstd->sp2, ret->lastp; means into s* accumulators
    for lo in range(0, N, B):
        hi = min(lo+B, N)
        nb_ = nn[lo:hi]
        mt = st[lo:hi]/nb_; vt = np.maximum(st2[lo:hi]/nb_ - mt**2, 0)
        mp = sp[lo:hi]/nb_; vp = np.maximum(sp2[lo:hi]/nb_ - mp**2, 0)
        stp[lo:hi] = (stp[lo:hi]/nb_ - mt*mp)/np.maximum(vt, 1e-9)      # slope
        sp2[lo:hi] = np.sqrt(vp)/np.maximum(np.abs(mp),1e-9)            # pstd
        lastp[lo:hi] = (lastp[lo:hi]-firstp[lo:hi])/np.maximum(firstp[lo:hi],1e-9)  # ret
        sspread[lo:hi] /= nb_; srel[lo:hi] /= nb_
        simb1[lo:hi] /= nb_; simb2[lo:hi] /= nb_; smicro[lo:hi] /= nb_
    slope = stp; pstd = sp2; ret = lastp
    spreadm = sspread; relm = srel; imb1m = simb1; imb2m = simb2; microm = smicro
    del st, st2, sp, firstp, maxsbp, minsbp
    gc.collect(); _libc.malloc_trim(0)
    feats = {}
    nb_ = n.reshape(ns, NB)
    total_bars = nb_.sum(1).astype(f64)
    frac = nb_/np.maximum(total_bars[:,None],1)
    for bi in range(NB):
        f = {}
        f[f'c{bi}_slope'] = slope[bi::NB]
        f[f'c{bi}_pstd'] = pstd[bi::NB]
        f[f'c{bi}_spread'] = spreadm[bi::NB]
        f[f'c{bi}_relspread'] = relm[bi::NB]
        f[f'c{bi}_imb1'] = imb1m[bi::NB]
        f[f'c{bi}_imb2'] = imb2m[bi::NB]
        f[f'c{bi}_micro'] = microm[bi::NB]
        f[f'c{bi}_maxtv'] = maxtv[bi::NB]
        f[f'c{bi}_barfrac'] = frac[:, bi]
        f[f'c{bi}_ret'] = ret[bi::NB]
        empty = (n[bi::NB]==0)
        for k, v in f.items():
            vv = v.astype(np.float32)
            vv[empty] = 0.0
            feats[k] = vv
    names = sorted(feats.keys())
    X = np.empty((ns, len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = feats[k]
    np.save(f'/tmp/work/X10_{split}.npy', X)
    np.save(f'/tmp/work/X10_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX'] = '1'
    split = sys.argv[1]; ns = int(sys.argv[2])
    build(split, ns)
