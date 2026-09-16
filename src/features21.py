# X21: microstructure v3-inspired gap filler. EARLY fine windows [0,10),[10,30),[30,60) sbp:
#  market: microprice mean, spread mean, L1 depth imbalance mean, mid last-first return
#  transaction: signed vol, signed amount, total vol, n
#  order: signed new pressure (new_buy-new_sell), cancel imbalance (can_sell-can_buy), net placement per side
# Reimplemented from public description of UnseenAnchor feat_microstructure_v3 (no code copied).
import sys, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
NB = 3
edges = np.array([10.0, 30.0])

def build(split, ns):
    t0 = time.time(); N = ns*NB; f64 = np.float64
    # ---- market pass ----
    n   = np.zeros(N, f64)
    smicro = np.zeros(N, f64); sspread = np.zeros(N, f64); simb = np.zeros(N, f64)
    smid_f = np.zeros(N, f64); smid_l = np.zeros(N, f64)  # approx first/last via min/max sbp is costly; use sum for mean only
    for ch in ziter(f'/tmp/mscapital/{split}/market.feather',
            ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1'], elems=1<<19):
        sid, sbp, a1, b1, av1, bv1 = ch
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        a1f = a1.astype(f64); b1f = b1.astype(f64)
        av = av1.astype(f64); bv = bv1.astype(f64)
        mid = (a1f+b1f)/2
        micro = (a1f*bv + b1f*av)/np.maximum(av+bv, 1e-9)
        spread = (a1f-b1f)/np.maximum(mid, 1e-9)
        imb = (bv-av)/np.maximum(av+bv, 1e-9)
        n += np.bincount(lin, minlength=N)
        smicro += np.bincount(lin, weights=micro, minlength=N)
        sspread += np.bincount(lin, weights=spread, minlength=N)
        simb += np.bincount(lin, weights=imb, minlength=N)
        smid_f += np.bincount(lin, weights=mid, minlength=N)
        _libc.malloc_trim(0)
    print(split, 'market pass', round(time.time()-t0), 's', flush=True)
    # ---- transaction pass ----
    tn = np.zeros(N, f64); tsvol = np.zeros(N, f64); tsamt = np.zeros(N, f64); ttot = np.zeros(N, f64)
    for sid, sbp, pr, v, side in ziter(f'/tmp/mscapital/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        vf = v.astype(f64); pf = pr.astype(f64)
        sgn = np.where(side==0, 1.0, -1.0)
        tn += np.bincount(lin, minlength=N)
        tsvol += np.bincount(lin, weights=vf*sgn, minlength=N)
        tsamt += np.bincount(lin, weights=pf*vf*sgn, minlength=N)
        ttot += np.bincount(lin, weights=vf, minlength=N)
        _libc.malloc_trim(0)
    print(split, 'tx pass', round(time.time()-t0), 's', flush=True)
    # ---- order pass ----
    onb = np.zeros(N, f64); ons = np.zeros(N, f64)  # new vol buy/sell
    ocb = np.zeros(N, f64); ocs = np.zeros(N, f64)  # cancel vol buy/sell
    for sid, sbp, pr, v, side, act in ziter(f'/tmp/mscapital/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action'], elems=1<<19):
        b = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*NB + b
        vf = v.astype(f64)
        is_new = (act==0); buy = (side==0)
        onb += np.bincount(lin, weights=np.where(is_new & buy, vf, 0.0), minlength=N)
        ons += np.bincount(lin, weights=np.where(is_new & ~buy, vf, 0.0), minlength=N)
        ocb += np.bincount(lin, weights=np.where(~is_new & buy, vf, 0.0), minlength=N)
        ocs += np.bincount(lin, weights=np.where(~is_new & ~buy, vf, 0.0), minlength=N)
        _libc.malloc_trim(0)
    print(split, 'order pass', round(time.time()-t0), 's', flush=True)
    # ---- assemble ----
    names = []; cols = []
    for bi in range(NB):
        sl = np.s_[bi::NB]
        nn = np.maximum(n[sl], 1); tt = np.maximum(ttot[sl], 1e-9)
        midm = smid_f[sl]/nn
        feats = {
          f'x21_m{bi}_micro_dev': np.where(n[sl]>0, smicro[sl]/np.maximum(smid_f[sl],1e-9)-1, 0.0),
          f'x21_m{bi}_spread': sspread[sl]/nn,
          f'x21_m{bi}_imb1': simb[sl]/nn,
          f'x21_t{bi}_signvol': tsvol[sl],
          f'x21_t{bi}_signvol_frac': tsvol[sl]/tt,
          f'x21_t{bi}_signamt': tsamt[sl]/np.maximum(midm,1e-9),
          f'x21_t{bi}_vol': ttot[sl],
          f'x21_o{bi}_newpress': (onb[sl]-ons[sl]),
          f'x21_o{bi}_canimb': (ocs[sl]-ocb[sl])/np.maximum(ocs[sl]+ocb[sl],1e-9),
          f'x21_o{bi}_netplace': (onb[sl]+ons[sl])-(ocb[sl]+ocs[sl]),
        }
        for k, vv in feats.items():
            names.append(k); cols.append(vv.astype(np.float32))
    X = np.stack(cols, axis=1)
    np.save(f'/tmp/work/X21_{split}.npy', X); np.save(f'/tmp/work/X21_{split}_names.npy', np.array(names))
    print(split, 'saved', X.shape, flush=True)

if __name__ == '__main__':
    import os
    os.environ['MALLOC_ARENA_MAX']='1'
    build(sys.argv[1], int(sys.argv[2]))
