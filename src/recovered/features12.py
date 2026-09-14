# X12: order price geography - volume-weighted distance from bucket mean mid, per (bucket, new/can, side).
# Pass1: bucket mean mid from market.feather. Pass2: order.feather stream.
import sys, numpy as np, time, ctypes, gc
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
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
    CL = 4  # new-buy, new-sell, can-buy, can-sell
    sw = np.zeros(N*CL, f64); swd = np.zeros(N*CL, f64); swa = np.zeros(N*CL, f64)
    for sid, sbp, pr, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        mref = bmid[lin]
        miss = nm_[lin]==0
        if miss.any():
            mref = mref.copy(); mref[miss] = gmean[sid.astype(np.int64)[miss]]
        d = (pr.astype(f64)-mref)/np.maximum(mref,1e-9)
        vf = v.astype(f64)
        cls = (np.where(act==0,0,2) + np.where(side==0,0,1)).astype(np.int64)
        lin4 = lin*CL + cls
        sw += np.bincount(lin4, weights=vf, minlength=N*CL)
        swd += np.bincount(lin4, weights=vf*d, minlength=N*CL)
        swa += np.bincount(lin4, weights=vf*np.abs(d), minlength=N*CL)
        _libc.malloc_trim(0)
    print(split, 'pass2 done', round(time.time()-t0), 's', flush=True)
    names=[]; cols=[]
    cname = ('nb','ns','cb','cs')
    for bi in range(NB):
        for ci, cn in enumerate(cname):
            sl = np.s_[(bi*CL+ci)::NB*CL]
            w = sw[sl]
            md = np.where(w>0, swd[sl]/np.maximum(w,1e-9), 0.0)
            ma = np.where(w>0, swa[sl]/np.maximum(w,1e-9), 0.0)
            for k, vv in ((f'g{bi}_{cn}_dist', md), (f'g{bi}_{cn}_absdist', ma)):
                vv = vv.astype(np.float32); vv[w==0] = 0.0
                names.append(k); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X12_{split}.npy', X); np.save(f'/tmp/work/X12_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
