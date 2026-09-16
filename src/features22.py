# X22: early-window dynamics + interactions, extends X21 (confirmed +0.0026/+0.0029/+0.0010).
# Early buckets E=[0,10),[10,30),[30,60); late ref L=[300,600).
#  market pass: book slope (a2-a1)/mid,(b1-b2)/mid per early bucket (6); imb1 & spread & microdev means per early+late (12)
#  tx pass: signed vol per early bucket (3)
#  order pass: new/can vol per side per early bucket -> cancel/new ratios + signed net per bucket (9+3)
#  assembled: deltas between windows + products (signvol*imb) + ratios. ~30 cols.
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NE = 3; edges_e = np.array([10.0, 30.0])
def build(split, ns):
    t0 = time.time(); f64 = np.float64
    NE4 = NE+1; N = ns*NE4  # bucket 3 = [300,600) late ref (from 4-way edges incl 300)
    edges4 = np.array([10.0, 30.0, 60.0, 300.0])  # 5 buckets: 0-10,10-30,30-60,60-300,300-600 ; use 0,1,2 and 4
    N5 = ns*5
    nm = np.zeros(N5, f64); ssa = np.zeros(N5, f64); ssb = np.zeros(N5, f64)
    simb = np.zeros(N5, f64); sspread = np.zeros(N5, f64); smicro = np.zeros(N5, f64); smid = np.zeros(N5, f64)
    for sid, sbp, a1, b1, av1, bv1, a2, b2 in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1','ask_price_2','bid_price_2'], elems=1<<19):
        b = np.searchsorted(edges4, sbp.astype(f64))
        lin = sid.astype(np.int64)*5 + b
        a1f=a1.astype(f64); b1f=b1.astype(f64); av=av1.astype(f64); bv=bv1.astype(f64)
        mid = (a1f+b1f)/2; m = np.maximum(mid,1e-9)
        nm += np.bincount(lin, minlength=N5)
        ssa += np.bincount(lin, weights=(a2.astype(f64)-a1f)/m, minlength=N5)
        ssb += np.bincount(lin, weights=(b1f-b2.astype(f64))/m, minlength=N5)
        simb += np.bincount(lin, weights=(bv-av)/np.maximum(av+bv,1e-9), minlength=N5)
        sspread += np.bincount(lin, weights=(a1f-b1f)/m, minlength=N5)
        smicro += np.bincount(lin, weights=(a1f*bv+b1f*av)/np.maximum(av+bv,1e-9), minlength=N5)
        smid += np.bincount(lin, weights=mid, minlength=N5)
        _libc.malloc_trim(0)
    print(split,'market',round(time.time()-t0),'s',flush=True)
    tsv = np.zeros(N5, f64)
    for sid, sbp, pr, v, side in ziter(f'/tmp/mscapital/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side'], elems=1<<19):
        b = np.searchsorted(edges4, sbp.astype(f64))
        lin = sid.astype(np.int64)*5 + b
        sgn = np.where(side==0,1.0,-1.0)
        tsv += np.bincount(lin, weights=v.astype(f64)*sgn, minlength=N5)
        _libc.malloc_trim(0)
    print(split,'tx',round(time.time()-t0),'s',flush=True)
    onb = np.zeros(N5, f64); ons = np.zeros(N5, f64); ocb = np.zeros(N5, f64); ocs = np.zeros(N5, f64)
    for sid, sbp, pr, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges4, sbp.astype(f64))
        lin = sid.astype(np.int64)*5 + b
        vf = v.astype(f64); is_new = (act==0); buy = (side==0)
        onb += np.bincount(lin, weights=np.where(is_new&buy, vf,0.0), minlength=N5)
        ons += np.bincount(lin, weights=np.where(is_new&~buy, vf,0.0), minlength=N5)
        ocb += np.bincount(lin, weights=np.where(~is_new&buy, vf,0.0), minlength=N5)
        ocs += np.bincount(lin, weights=np.where(~is_new&~buy, vf,0.0), minlength=N5)
        _libc.malloc_trim(0)
    print(split,'order',round(time.time()-t0),'s',flush=True)
    names=[]; cols=[]
    nn = np.maximum(nm,1)
    imb = simb/nn; spread = sspread/nn; microdev = np.where(nm>0, smicro/np.maximum(smid,1e-9)-1, 0.0)
    slopea = ssa/nn; slopeb = ssb/nn
    E = [0,1,2]; LATE = 4
    for bi in E:
        sl = np.s_[bi::5]
        names += [f'x22_sla{bi}', f'x22_slb{bi}']; cols += [slopea[sl].astype(np.float32), slopeb[sl].astype(np.float32)]
        names.append(f'x22_sv{bi}'); cols.append(tsv[sl].astype(np.float32))
        # interactions
        names.append(f'x22_svXimb{bi}'); cols.append((tsv[sl]*imb[sl]).astype(np.float32))
        # cancel/new ratios per side
        names.append(f'x22_cnrb{bi}'); cols.append((ocb[sl]/np.maximum(onb[sl],1e-9)).astype(np.float32))
        names.append(f'x22_cnrs{bi}'); cols.append((ocs[sl]/np.maximum(ons[sl],1e-9)).astype(np.float32))
        # signed net placement
        names.append(f'x22_net{bi}'); cols.append(((onb[sl]-ocb[sl])-(ons[sl]-ocs[sl])).astype(np.float32))
    # deltas between windows
    def S(bi): return np.s_[bi::5]
    names.append('x22_d_sv_01');  cols.append((tsv[S(0)]-tsv[S(1)]).astype(np.float32))
    names.append('x22_d_sv_12');  cols.append((tsv[S(1)]-tsv[S(2)]).astype(np.float32))
    names.append('x22_d_imb_02'); cols.append((imb[S(0)]-imb[S(2)]).astype(np.float32))
    names.append('x22_d_imb_0L'); cols.append((imb[S(0)]-imb[S(LATE)]).astype(np.float32))
    names.append('x22_d_spr_0L'); cols.append((spread[S(0)]-spread[S(LATE)]).astype(np.float32))
    names.append('x22_d_mic_0L'); cols.append((microdev[S(0)]-microdev[S(LATE)]).astype(np.float32))
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X22_{split}.npy', X); np.save(f'/tmp/work/X22_{split}_names.npy', np.array(names))
    print(split,'saved',X.shape,flush=True)
if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
