# X8: transaction flow bucketed by sbp. Buckets [0,60),[60,180),[180,300),[300,600).
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
def build(split, ns):
    t0=time.time(); N=ns*NB
    n=np.zeros(N,np.float64); sv=np.zeros(N,np.float64); sbv=np.zeros(N,np.float64)
    spv=np.zeros(N,np.float64); sp=np.zeros(N,np.float64); mxv=np.zeros(N,np.float64)
    for sid, sbp, price, v, side in ziter(f'/tmp/mscapital/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(np.float64))
        lin = sid.astype(np.int64)*NB + b
        vf = v.astype(np.float64); pf = price.astype(np.float64)
        buy = (side==0)
        n += np.bincount(lin, minlength=N)
        sv += np.bincount(lin, weights=vf, minlength=N)
        sbv += np.bincount(lin, weights=np.where(buy, vf, 0.0), minlength=N)
        spv += np.bincount(lin, weights=pf*vf, minlength=N)
        sp += np.bincount(lin, weights=pf, minlength=N)
        np.maximum.at(mxv, lin, vf)
        _libc.malloc_trim(0)
    print(split,'stream',round(time.time()-t0),'s',flush=True)
    nn = np.maximum(n,1)
    names=[]; cols=[]
    for bi in range(NB):
        sl = np.s_[bi::NB]
        tot = sv[sl]; cnt = n[sl]
        feats = {
          f't{bi}_n': cnt,
          f't{bi}_vol': tot,
          f't{bi}_buyfrac': np.where(tot>0, sbv[sl]/np.maximum(tot,1e-9), 0.5),
          f't{bi}_maxv': mxv[sl],
          f't{bi}_vwap_dev': np.where(tot>0, (spv[sl]/np.maximum(tot,1e-9))/np.maximum(sp[sl]/nn[sl],1e-9)-1, 0.0),
          f't{bi}_avgsz': np.where(cnt>0, tot/nn[sl], 0.0),
        }
        empty = cnt==0
        for k,vv in feats.items():
            vv = vv.astype(np.float32); vv[empty]=0.0
            names.append(k); cols.append(vv)
    X8 = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X8_{split}.npy', X8); np.save(f'/tmp/work/X8_{split}_names.npy', np.array(names))
    print(split,'X8 saved',X8.shape,round(time.time()-t0),'s',flush=True)
if __name__=='__main__':
    build('train',1257637); gc.collect(); _libc.malloc_trim(0)
    build('test',647896)
    print('X8_DONE',flush=True)
