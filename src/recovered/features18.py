# X18: net placement (new minus cancel volume) and buy-sell placement imbalance in NEAR vs FAR zones per bucket.
# near: |d| < 5e-4, far: >= 5e-4. Per bucket: near_net, far_net, near_buyfrac_new, far_buyfrac_new, near_cancel_imb, far_cancel_imb = 24.
import sys, numpy as np, time, ctypes, gc
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
DTH = 5e-4
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
    # per (bucket, zone near/far): snew, scan, sbuynew, sbuycan, stot
    CL = 2
    snew = np.zeros(N*CL, f64); scan = np.zeros(N*CL, f64)
    sbn = np.zeros(N*CL, f64); sbc = np.zeros(N*CL, f64); stot = np.zeros(N*CL, f64)
    for sid, sbp, pr, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        mref = bmid[lin]
        miss = nm_[lin]==0
        if miss.any():
            mref = mref.copy(); mref[miss] = gmean[sid.astype(np.int64)[miss]]
        ad = np.abs(pr.astype(f64)-mref)/np.maximum(mref,1e-9)
        z = (ad >= DTH).astype(np.int64)
        vf = v.astype(f64)
        new = (act==0); buy = (side==0)
        lin2 = lin*CL + z
        snew += np.bincount(lin2, weights=np.where(new, vf, 0.0), minlength=N*CL)
        scan += np.bincount(lin2, weights=np.where(~new, vf, 0.0), minlength=N*CL)
        sbn += np.bincount(lin2, weights=np.where(new&buy, vf, 0.0), minlength=N*CL)
        sbc += np.bincount(lin2, weights=np.where((~new)&buy, vf, 0.0), minlength=N*CL)
        stot += np.bincount(lin2, weights=vf, minlength=N*CL)
        _libc.malloc_trim(0)
    print(split, 'pass2 done', round(time.time()-t0), 's', flush=True)
    names=[]; cols=[]
    zn = ('near','far')
    for bi in range(NB):
        for zi in range(2):
            sl = np.s_[(bi*CL+zi)::NB*CL]
            tot = stot[sl]; e = tot==0
            f1 = (snew[sl]-scan[sl])/np.maximum(tot,1e-9)              # net placement
            f2 = np.where(snew[sl]>0, sbn[sl]/np.maximum(snew[sl],1e-9), 0.5)  # buyfrac new
            f3 = np.where(scan[sl]>0, sbc[sl]/np.maximum(scan[sl],1e-9), 0.5)  # buyfrac can
            for k, vv in ((f'z{bi}_{zn[zi]}_net', f1), (f'z{bi}_{zn[zi]}_bfnew', f2), (f'z{bi}_{zn[zi]}_bfcan', f3)):
                vv = vv.astype(np.float32); vv[e] = 0.0
                names.append(k); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X18_{split}.npy', X); np.save(f'/tmp/work/X18_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
