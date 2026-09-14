# X7: L2 book + price features bucketed by seconds-before-predict.
# Buckets: [0,60), [60,180), [180,300), [300,600). ~9 stats per bucket -> 36 feats + global bar count.
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
def bucket(sbp):
    return np.searchsorted(edges, sbp)
def build(split, ns):
    t0 = time.time()
    N = ns*NB
    n   = np.zeros(N, np.int64)
    st  = np.zeros(N, np.float64); st2 = np.zeros(N, np.float64)
    sp  = np.zeros(N, np.float64); sp2 = np.zeros(N, np.float64)
    stp = np.zeros(N, np.float64)
    sspread = np.zeros(N, np.float64); srel = np.zeros(N, np.float64)
    simb1 = np.zeros(N, np.float64); simb2 = np.zeros(N, np.float64)
    smicro = np.zeros(N, np.float64)
    maxtv = np.zeros(N, np.float64)
    cols = ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume',
            'ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
            'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2']
    for ch in ziter(f'/tmp/mscapital/{split}/market.feather', cols, elems=1<<19):
        (sid, sbp, ap, tv, a1, b1, av1, bv1, a2, b2, av2, bv2) = ch
        b = bucket(sbp.astype(np.float64))
        lin = sid.astype(np.int64)*NB + b
        t = sbp.astype(np.float64); x = ap.astype(np.float64)
        a1f = a1.astype(np.float64); b1f = b1.astype(np.float64)
        mid = (a1f+b1f)/2
        n   += np.bincount(lin, minlength=N)
        st  += np.bincount(lin, weights=t, minlength=N)
        st2 += np.bincount(lin, weights=t*t, minlength=N)
        sp  += np.bincount(lin, weights=x, minlength=N)
        sp2 += np.bincount(lin, weights=x*x, minlength=N)
        stp += np.bincount(lin, weights=t*x, minlength=N)
        sspread += np.bincount(lin, weights=(a1f-b1f), minlength=N)
        srel += np.bincount(lin, weights=(a1f-b1f)/np.maximum(mid,1e-9), minlength=N)
        imb1 = (bv1.astype(np.float64)-av1)/(bv1.astype(np.float64)+av1+1)
        bvv = bv1.astype(np.float64)+bv2; avv = av1.astype(np.float64)+av2
        imb2 = (bvv-avv)/(bvv+avv+1)
        simb1 += np.bincount(lin, weights=imb1, minlength=N)
        simb2 += np.bincount(lin, weights=imb2, minlength=N)
        denom = np.maximum(av1.astype(np.float64)+bv1, 1)
        micro = (a1f*bv1 + b1f*av1)/denom
        smicro += np.bincount(lin, weights=micro/np.maximum(mid,1e-9)-1, minlength=N)
        # bucket volume proxy: max of cumulative tv within (sid,bucket)
        np.maximum.at(maxtv, lin, tv.astype(np.float64))
        _libc.malloc_trim(0)
    print(split, 'stream done', round(time.time()-t0), 's', flush=True)
    # derive per-bucket features
    nn = np.maximum(n, 1).astype(np.float64)
    mean_t = st/nn; var_t = np.maximum(st2/nn - mean_t**2, 0)
    mean_p = sp/nn; var_p = np.maximum(sp2/nn - mean_p**2, 0)
    cov = stp/nn - mean_t*mean_p
    slope = cov/np.maximum(var_t, 1e-9)
    pstd = np.sqrt(var_p)/np.maximum(np.abs(mean_p),1e-9)
    feats = {}
    nb_ = n.reshape(ns, NB)
    total_bars = nb_.sum(1).astype(np.float64)
    frac = nb_/np.maximum(total_bars[:,None],1)
    for bi in range(NB):
        sl = slice(bi, None, NB)
        f = {}
        f[f'b{bi}_slope'] = slope[bi::NB]
        f[f'b{bi}_pstd'] = pstd[bi::NB]
        f[f'b{bi}_spread'] = (sspread/nn)[bi::NB]
        f[f'b{bi}_relspread'] = (srel/nn)[bi::NB]
        f[f'b{bi}_imb1'] = (simb1/nn)[bi::NB]
        f[f'b{bi}_imb2'] = (simb2/nn)[bi::NB]
        f[f'b{bi}_micro'] = (smicro/nn)[bi::NB]
        f[f'b{bi}_maxtv'] = maxtv[bi::NB]
        f[f'b{bi}_barfrac'] = frac[:, bi]
        for k, v in f.items():
            feats[k] = v.astype(np.float32)
        # mask empty buckets
        empty = (n[bi::NB]==0)
        for k in f:
            feats[k][empty] = 0.0
    names = list(feats.keys())
    X7 = np.stack([feats[k] for k in names], axis=1)
    np.save(f'/tmp/work/X7_{split}.npy', X7)
    np.save(f'/tmp/work/X7_{split}_names.npy', np.array(names))
    print(split, 'X7 saved', X7.shape, round(time.time()-t0), 's', flush=True)
if __name__ == '__main__':
    build('train', 1257637)
    gc.collect(); _libc.malloc_trim(0)
    build('test', 647896)
    print('X7_DONE', flush=True)
