# X19: fill-ratio family - links order placements to executions per (sid, sbp-bucket, side).
# placed/canceled vol from order.feather, executed vol from transaction.feather. GBM cannot divide across streams.
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
def build(split, ns):
    t0=time.time(); N=ns*NB*2  # bucket x side
    pvol=np.zeros(N); cvol=np.zeros(N); evol=np.zeros(N); evolw=np.zeros(N)
    for sid, sbp, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(np.float64))
        lin = (sid.astype(np.int64)*NB + b)*2 + (side!=0).astype(np.int64)
        vf = v.astype(np.float64); new = (act==0)
        pvol += np.bincount(lin, weights=np.where(new, vf, 0.0), minlength=N)
        cvol += np.bincount(lin, weights=np.where(~new, vf, 0.0), minlength=N)
        _libc.malloc_trim(0)
    print(split,'orders pass',round(time.time()-t0),'s',flush=True)
    for sid, sbp, pr, v, side in ziter(f'/tmp/mscapital/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(np.float64))
        lin = (sid.astype(np.int64)*NB + b)*2 + (side!=0).astype(np.int64)
        vf = v.astype(np.float64)
        evol += np.bincount(lin, weights=vf, minlength=N)
        _libc.malloc_trim(0)
    print(split,'trades pass',round(time.time()-t0),'s',flush=True)
    names=[]; cols=[]
    sn=('b','s')
    for bi in range(NB):
        for si in range(2):
            sl = np.s_[(bi*2+si)::NB*2]
            pv=pvol[sl]; cv=cvol[sl]; ev=evol[sl]
            net = pv-cv
            empty = (pv+ev)==0
            feats = {
              f'f{bi}{sn[si]}_fill': ev/np.maximum(pv,1.0),
              f'f{bi}{sn[si]}_fillnet': ev/np.maximum(net,1.0),
              f'f{bi}{sn[si]}_exec_logratio': np.log1p(ev)-np.log1p(pv),
              f'f{bi}{sn[si]}_cancelover': cv/np.maximum(pv,1.0),
            }
            for k,vv in feats.items():
                vv = vv.astype(np.float32)
                vv[~np.isfinite(vv)]=0.0; vv[empty]=0.0
                names.append(k); cols.append(vv)
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X19_{split}.npy', X); np.save(f'/tmp/work/X19_{split}_names.npy', np.array(names))
    print(split,'saved',X.shape,flush=True)
if __name__=='__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
