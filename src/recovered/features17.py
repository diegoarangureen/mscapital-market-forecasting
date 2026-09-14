# X17: whale-conditioned TRADE geography - trade dist/absdist from bucket mid, split by trade size vs (sid,bucket) mean. 4 buckets x (small/big) x (buy/sell): dist+absdist = 16. Plus signed: buy-big minus sell-big dist = 4. Total 20.
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
    # pass2: mean trade volume per (sid,bucket)
    sv = np.zeros(N, f64); nv = np.zeros(N, f64)
    for sid, sbp, pr, v, side in ziter(f'/tmp/mscapital/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        sv += np.bincount(lin, weights=v.astype(f64), minlength=N)
        nv += np.bincount(lin, minlength=N)
        _libc.malloc_trim(0)
    meanv = sv/np.maximum(nv,1)
    del sv, nv; gc.collect(); _libc.malloc_trim(0)
    print(split, 'pass2 done', round(time.time()-t0), 's', flush=True)
    CL = 4  # (small/big) x (buy/sell)
    sw = np.zeros(N*CL, f64); swd = np.zeros(N*CL, f64); swa = np.zeros(N*CL, f64)
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
        big = (vf > meanv[lin]).astype(np.int64)
        lin4 = lin*CL + big*2 + (side!=0).astype(np.int64)
        sw += np.bincount(lin4, weights=vf, minlength=N*CL)
        swd += np.bincount(lin4, weights=vf*d, minlength=N*CL)
        swa += np.bincount(lin4, weights=vf*np.abs(d), minlength=N*CL)
        _libc.malloc_trim(0)
    print(split, 'pass3 done', round(time.time()-t0), 's', flush=True)
    names=[]; cols=[]
    gn_=('sm','big'); sn_=('b','s')
    md = {}
    for bi in range(NB):
        for gi in range(2):
            for si in range(2):
                sl = np.s_[(bi*CL+gi*2+si)::NB*CL]
                w = sw[sl]
                dmean = np.where(w>0, swd[sl]/np.maximum(w,1e-9), 0.0)
                amean = np.where(w>0, swa[sl]/np.maximum(w,1e-9), 0.0)
                for stat, arr in (('dist', dmean), ('absdist', amean)):
                    vv = arr.astype(np.float32); vv[w==0] = 0.0
                    names.append(f'tw{bi}_{gn_[gi]}{sn_[si]}_{stat}'); cols.append(vv)
                md[(bi,gi,si)] = dmean
    for bi in range(NB):
        vv = (md[(bi,1,0)] - md[(bi,1,1)]).astype(np.float32)
        names.append(f'tw{bi}_bigbs_ddist'); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X17_{split}.npy', X); np.save(f'/tmp/work/X17_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
