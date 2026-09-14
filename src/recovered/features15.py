# X15: book level spacing per sbp bucket. (a2-a1)/mid, (b1-b2)/mid, L1 vol share, L1-L2 vol gap. 4x4=16.
import sys, numpy as np, time, ctypes, gc
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
def build(split, ns):
    t0 = time.time(); N = ns*NB; f64 = np.float64
    n = np.zeros(N, f64)
    sspa = np.zeros(N, f64); sspb = np.zeros(N, f64)
    svs = np.zeros(N, f64); svg = np.zeros(N, f64)
    for sid, sbp, a1, b1, av1, bv1, a2, b2, av2, bv2 in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
             'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        mid = (a1.astype(f64)+b1.astype(f64))/2
        n += np.bincount(lin, minlength=N)
        sspa += np.bincount(lin, weights=(a2.astype(f64)-a1)/np.maximum(mid,1e-9), minlength=N)
        sspb += np.bincount(lin, weights=(b1.astype(f64)-b2)/np.maximum(mid,1e-9), minlength=N)
        av1f = av1.astype(f64); bv1f = bv1.astype(f64); av2f = av2.astype(f64); bv2f = bv2.astype(f64)
        svs += np.bincount(lin, weights=(av1f+bv1f)/np.maximum(av1f+bv1f+av2f+bv2f,1e-9), minlength=N)
        svg += np.bincount(lin, weights=((av1f+bv1f)-(av2f+bv2f))/np.maximum(av1f+bv1f+av2f+bv2f,1e-9), minlength=N)
        _libc.malloc_trim(0)
    print(split, 'stream done', round(time.time()-t0), 's', flush=True)
    nn = np.maximum(n, 1)
    feats = {}
    for bi in range(NB):
        sl = np.s_[bi::NB]
        e = n[sl]==0
        for k, vv in ((f's{bi}_askgap', sspa[sl]/nn[sl]), (f's{bi}_bidgap', sspb[sl]/nn[sl]),
                      (f's{bi}_l1vshare', svs[sl]/nn[sl]), (f's{bi}_l12volgap', svg[sl]/nn[sl])):
            vv = vv.astype(np.float32); vv[e] = 0.0
            feats[k] = vv
    names = sorted(feats.keys())
    X = np.empty((ns, len(names)), np.float32)
    for j,k in enumerate(names): X[:,j] = feats[k]
    np.save(f'/tmp/work/X15_{split}.npy', X); np.save(f'/tmp/work/X15_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
