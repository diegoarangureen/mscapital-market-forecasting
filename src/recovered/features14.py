# X14: order volume share across distance-from-mid bands, per (bucket, new/can). 4 bands x 2 actions x 4 buckets = 32.
# Bands on |d| relative: [0,2e-4), [2e-4,1e-3), [1e-3,5e-3), [5e-3,inf).
import sys, numpy as np, time, ctypes, gc
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4; NBAND = 4
edges = np.array([60.0, 180.0, 300.0])
bedges = np.array([2e-4, 1e-3, 5e-3])
def build(split, ns):
    t0 = time.time(); N = ns*NB; f64 = np.float64
    nm_ = np.zeros(N, f64); smid = np.zeros(N, f64)
    gn = np.zeros(ns, f64); gmid = np.zeros(ns, f64)
    for sid, sbp, a1, b1 in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','ask_price_1','bid_price_1'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        mid = (a1.astype(f64)+b1.astype(f64))/2
        nm_ += np.bincount(lin, minlength=N)
        smid += np.bincount(lin, weights=mid, minlength=N)
        gn += np.bincount(sid.astype(np.int64), minlength=ns)
        gmid += np.bincount(sid.astype(np.int64), weights=mid, minlength=ns)
        _libc.malloc_trim(0)
    bmid = np.where(nm_>0, smid/np.maximum(nm_,1), 0.0)
    gmean = gmid/np.maximum(gn,1)
    del smid; gc.collect(); _libc.malloc_trim(0)
    print(split, 'pass1 done', round(time.time()-t0), 's', flush=True)
    CL = 2*NBAND  # (new/can) x 4 bands
    sw = np.zeros(N*CL, f64)
    stot = np.zeros(N*2, f64)
    for sid, sbp, pr, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        mref = bmid[lin]
        miss = nm_[lin]==0
        if miss.any():
            mref = mref.copy(); mref[miss] = gmean[sid.astype(np.int64)[miss]]
        ad = np.abs(pr.astype(f64)-mref)/np.maximum(mref,1e-9)
        band = np.searchsorted(bedges, ad)
        vf = v.astype(f64)
        a = (act!=0).astype(np.int64)
        linb = lin*CL + a*NBAND + band
        sw += np.bincount(linb, weights=vf, minlength=N*CL)
        lin2 = lin*2 + a
        stot += np.bincount(lin2, weights=vf, minlength=N*2)
        _libc.malloc_trim(0)
    print(split, 'pass2 done', round(time.time()-t0), 's', flush=True)
    names=[]; cols=[]
    an = ('new','can')
    for bi in range(NB):
        for ai in range(2):
            tot = stot[(bi*2+ai)::NB*2]
            for band_i in range(NBAND):
                sl = np.s_[(bi*CL+ai*NBAND+band_i)::NB*CL]
                vv = (sw[sl]/np.maximum(tot,1e-9)).astype(np.float32)
                vv[tot==0] = 0.0
                names.append(f'bd{bi}_{an[ai]}_band{band_i}'); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X14_{split}.npy', X); np.save(f'/tmp/work/X14_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
