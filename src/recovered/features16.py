# X16: whale-conditioned order geography - dist/absdist vol-weighted, split by order size vs bucket median. 4 buckets x 2 sizes x (new/can) = 16*2=32... simplified: (bucket, new/can, small/big) mean dist = 16, mean absdist = 16.
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
    # pass2: per (sid,bucket) median volume -> need 2 passes over orders; approximate median with mean volume per (sid,bucket,act)
    CL2 = 2
    sv2 = np.zeros(N*CL2, f64); nv2 = np.zeros(N*CL2, f64)
    for sid, sbp, pr, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        a = (act!=0).astype(np.int64)
        lin2 = lin*CL2 + a
        sv2 += np.bincount(lin2, weights=v.astype(f64), minlength=N*CL2)
        nv2 += np.bincount(lin2, minlength=N*CL2)
        _libc.malloc_trim(0)
    meanv = sv2/np.maximum(nv2,1)
    del sv2, nv2; gc.collect(); _libc.malloc_trim(0)
    print(split, 'pass2 done', round(time.time()-t0), 's', flush=True)
    # pass3: dist stats split by v <= meanv (small) vs > (big)
    CL4 = 4  # (new/can) x (small/big)
    sw = np.zeros(N*CL4, f64); swd = np.zeros(N*CL4, f64); swa = np.zeros(N*CL4, f64)
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
        a = (act!=0).astype(np.int64)
        big = (vf > meanv[lin*CL2+a]).astype(np.int64)
        lin4 = lin*CL4 + a*2 + big
        sw += np.bincount(lin4, weights=vf, minlength=N*CL4)
        swd += np.bincount(lin4, weights=vf*d, minlength=N*CL4)
        swa += np.bincount(lin4, weights=vf*np.abs(d), minlength=N*CL4)
        _libc.malloc_trim(0)
    print(split, 'pass3 done', round(time.time()-t0), 's', flush=True)
    names=[]; cols=[]
    an=('new','can'); sn=('sm','big')
    for bi in range(NB):
        for ai in range(2):
            for gi in range(2):
                sl = np.s_[(bi*CL4+ai*2+gi)::NB*CL4]
                w = sw[sl]
                for stat, arr in (('dist', swd), ('absdist', swa)):
                    vv = np.where(w>0, arr[sl]/np.maximum(w,1e-9), 0.0).astype(np.float32)
                    vv[w==0] = 0.0
                    names.append(f'w{bi}_{an[ai]}_{sn[gi]}_{stat}'); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X16_{split}.npy', X); np.save(f'/tmp/work/X16_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
