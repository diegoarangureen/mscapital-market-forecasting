# X13: extends X12 family. (a) trade price geography: vol-weighted dist from bucket mid per (bucket, side) = 8+8.
# (b) order distance second moment (d^2 vol-weighted) per (bucket, new/can, side) = 16. Total 32.
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
    # (a) trades: per (bucket, side 0=buy/1=sell): sw, swd, swa
    CL = 2
    tw = np.zeros(N*CL, f64); twd = np.zeros(N*CL, f64); twa = np.zeros(N*CL, f64)
    for sid, sbp, pr, v, side in ziter(f'/tmp/mscapital/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        mref = bmid[lin]
        miss = nm_[lin]==0
        if miss.any():
            mref = mref.copy(); mref[miss] = gmean[sid.astype(np.int64)[miss]]
        d = (pr.astype(f64)-mref)/np.maximum(mref,1e-9)
        vf = v.astype(f64)
        lin2 = lin*CL + (side!=0).astype(np.int64)
        tw += np.bincount(lin2, weights=vf, minlength=N*CL)
        twd += np.bincount(lin2, weights=vf*d, minlength=N*CL)
        twa += np.bincount(lin2, weights=vf*np.abs(d), minlength=N*CL)
        _libc.malloc_trim(0)
    print(split, 'pass2 trades done', round(time.time()-t0), 's', flush=True)
    # (b) orders: second moment of dist per (bucket, new/can, side)
    CL4 = 4
    ow = np.zeros(N*CL4, f64); od2 = np.zeros(N*CL4, f64)
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
        lin4 = lin*CL4 + cls
        ow += np.bincount(lin4, weights=vf, minlength=N*CL4)
        od2 += np.bincount(lin4, weights=vf*d*d, minlength=N*CL4)
        _libc.malloc_trim(0)
    print(split, 'pass3 orders done', round(time.time()-t0), 's', flush=True)
    names=[]; cols=[]
    for bi in range(NB):
        for ci, cn in enumerate(('tb','ts')):
            sl = np.s_[(bi*CL+ci)::NB*CL]
            w = tw[sl]
            for k, vv in ((f't{bi}_{cn}_dist', np.where(w>0, twd[sl]/np.maximum(w,1e-9), 0.0)),
                          (f't{bi}_{cn}_absdist', np.where(w>0, twa[sl]/np.maximum(w,1e-9), 0.0))):
                vv = vv.astype(np.float32); vv[w==0] = 0.0
                names.append(k); cols.append(vv)
    for bi in range(NB):
        for ci, cn in enumerate(('nb','ns','cb','cs')):
            sl = np.s_[(bi*CL4+ci)::NB*CL4]
            w = ow[sl]
            vv = np.where(w>0, np.sqrt(np.maximum(od2[sl]/np.maximum(w,1e-9),0.0)), 0.0).astype(np.float32)
            vv[w==0] = 0.0
            names.append(f'g{bi}_{cn}_rmsdist'); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X13_{split}.npy', X); np.save(f'/tmp/work/X13_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
