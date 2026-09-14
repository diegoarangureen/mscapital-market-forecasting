# X9: order events bucketed by sbp. Same buckets. action 0=new else cancel-ish.
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 4
edges = np.array([60.0, 180.0, 300.0])
def build(split, ns):
    t0=time.time(); N=ns*NB
    nn_=np.zeros(N,np.float64); nc=np.zeros(N,np.float64)
    svn=np.zeros(N,np.float64); svc=np.zeros(N,np.float64)
    sbn=np.zeros(N,np.float64); sbc=np.zeros(N,np.float64)
    for sid, sbp, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(np.float64))
        lin = sid.astype(np.int64)*NB + b
        vf = v.astype(np.float64); new = (act==0); buy = (side==0)
        nn_ += np.bincount(lin, weights=new.astype(np.float64), minlength=N)
        nc += np.bincount(lin, weights=(~new).astype(np.float64), minlength=N)
        svn += np.bincount(lin, weights=np.where(new, vf, 0.0), minlength=N)
        svc += np.bincount(lin, weights=np.where(~new, vf, 0.0), minlength=N)
        sbn += np.bincount(lin, weights=np.where(new&buy, vf, 0.0), minlength=N)
        sbc += np.bincount(lin, weights=np.where((~new)&buy, vf, 0.0), minlength=N)
        _libc.malloc_trim(0)
    print(split,'stream',round(time.time()-t0),'s',flush=True)
    names=[]; cols=[]
    for bi in range(NB):
        sl = np.s_[bi::NB]
        feats = {
          f'o{bi}_nnew': nn_[sl], f'o{bi}_ncan': nc[sl],
          f'o{bi}_vnew': svn[sl], f'o{bi}_vcan': svc[sl],
          f'o{bi}_netc': nn_[sl]-nc[sl],
          f'o{bi}_buyfrac_new': np.where(svn[sl]>0, sbn[sl]/np.maximum(svn[sl],1e-9), 0.5),
          f'o{bi}_buyfrac_can': np.where(svc[sl]>0, sbc[sl]/np.maximum(svc[sl],1e-9), 0.5),
          f'o{bi}_canratio': nc[sl]/np.maximum(nn_[sl],1.0),
        }
        empty = (nn_[sl]+nc[sl])==0
        for k,vv in feats.items():
            vv = vv.astype(np.float32); vv[empty]=0.0
            names.append(k); cols.append(vv)
    X9 = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X9_{split}.npy', X9); np.save(f'/tmp/work/X9_{split}_names.npy', np.array(names))
    print(split,'X9 saved',X9.shape,round(time.time()-t0),'s',flush=True)
if __name__=='__main__':
    build('train',1257637); gc.collect(); _libc.malloc_trim(0)
    build('test',647896)
    print('X9_DONE',flush=True)
