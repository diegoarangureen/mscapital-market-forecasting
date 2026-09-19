# X25: trade arrival dynamics + price impact features (absolute, per-sample, no cross-sectional).
# Source: transaction.feather (60s window: sample_id, seconds_before_predict, price, volume, side).
# Families: inter-arrival stats, burstiness, intensity ramp (last 10s), Kyle lambda, price impact,
# trade-size moments, price-level churn, VWAP drift. Streaming bincount pattern (X22/X24 style).
import os, time
import numpy as np

DATA = os.environ.get('COMP', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = {'train': 1257637, 'test': 647896}
f64 = np.float64
edges = np.array([20.0, 40.0])  # buckets over the 60s trade window: [40,60),[20,40),[0,20) in sbp terms

def ziter(path, cols, elems=1<<19):
    import pyarrow as pa
    reader = pa.ipc.open_file(path)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        arrays = [b.column(c).to_numpy(zero_copy_only=False) for c in cols]
        n = len(arrays[0])
        for s in range(0, n, elems):
            yield tuple(a[s:s+elems] for a in arrays)

def build(split):
    ns = NS[split]; t0 = time.time()
    N3 = ns * 3
    n_t = np.zeros(N3, f64); tv = np.zeros(N3, f64)
    # inter-arrival accumulators (per sample, full window): rows sorted by (sid, sbp desc) -> dt = prev_sbp - sbp
    ia_sum = np.zeros(ns, f64); ia_sq = np.zeros(ns, f64); ia_max = np.zeros(ns, f64)
    prev_t = np.zeros(ns, f64); seen = np.zeros(ns, bool); cnt_ia = np.zeros(ns, f64)
    # kyle lambda / price impact (per sample): consecutive-trade price diffs
    kl_num = np.zeros(ns, f64); kl_den = np.zeros(ns, f64)
    pi_num = np.zeros(ns, f64); pi_den = np.zeros(ns, f64)
    n_chg = np.zeros(ns, f64)  # price changes between consecutive trades
    prev_p = np.zeros(ns, f64)
    # trade size moments
    v1 = np.zeros(ns, f64); v2 = np.zeros(ns, f64); v3 = np.zeros(ns, f64); vmax = np.zeros(ns, f64)
    # intensity ramp: last 10s (sbp<10) count/volume
    n_l10 = np.zeros(ns, f64); v_l10 = np.zeros(ns, f64)
    # vwap drift
    pv_sum = np.zeros(ns, f64); first_p = np.zeros(ns, f64)
    for sid, sbp, pr, v, side in ziter(f'{DATA}/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side']):
        s64 = sid.astype(np.int64); tf = sbp.astype(f64); pf = pr.astype(f64); vf = v.astype(f64)
        bkt = np.searchsorted(edges, tf)
        lin = s64*3 + bkt
        n_t += np.bincount(lin, minlength=N3)
        tv += np.bincount(lin, weights=vf, minlength=N3)
        # inter-arrival (per sample)
        pt = prev_t[s64]; has = seen[s64]
        dt = np.where(has, pt - tf, 0.0)  # sbp desc -> dt >= 0
        dt = np.maximum(dt, 0.0)
        ia_sum += np.bincount(s64, weights=np.where(has, dt, 0.0), minlength=ns)
        ia_sq  += np.bincount(s64, weights=np.where(has, dt*dt, 0.0), minlength=ns)
        np.maximum.at(ia_max, s64[has], dt[has])
        cnt_ia += np.bincount(s64[has], minlength=ns)
        prev_t[s64] = tf; seen[s64] = True
        # kyle lambda: dprice * signedvol ; impact: |dprice| * sqrt(vol)
        pp = prev_p[s64]
        sgn = np.where(side==0, 1.0, -1.0)
        dp = np.where(has, pf - pp, 0.0)
        sv = vf*sgn
        kl_num += np.bincount(s64, weights=dp*sv, minlength=ns)
        kl_den += np.bincount(s64, weights=sv*sv, minlength=ns)
        sq = np.sqrt(vf)
        pi_num += np.bincount(s64, weights=np.abs(dp)*sq, minlength=ns)
        pi_den += np.bincount(s64, weights=sq, minlength=ns)
        n_chg += np.bincount(s64[has & (dp != 0)], minlength=ns)
        prev_p[s64] = pf
        # size moments
        v1 += np.bincount(s64, weights=vf, minlength=ns)
        v2 += np.bincount(s64, weights=vf*vf, minlength=ns)
        v3 += np.bincount(s64, weights=vf*vf*vf, minlength=ns)
        np.maximum.at(vmax, s64, vf)
        # last 10s
        l10 = tf < 10.0
        n_l10 += np.bincount(s64[l10], minlength=ns)
        v_l10 += np.bincount(s64[l10], weights=vf[l10], minlength=ns)
        # vwap drift: first trade price = row with max sbp (first seen)
        pv_sum += np.bincount(s64, weights=pf*vf, minlength=ns)
    print(split, 'tx pass', round(time.time()-t0), 's', flush=True)

    nn = np.maximum(cnt_ia, 1.0)
    ia_mu = ia_sum/nn; ia_sd = np.sqrt(np.maximum(ia_sq/nn - ia_mu**2, 0))
    nfull = np.maximum(n_t.reshape(ns,3).sum(1), 1.0)
    vm = v1/nfull; vsd = np.sqrt(np.maximum(v2/nfull - vm**2, 0))
    vsk = np.where(vsd>0, (v3/nfull - 3*vm*vsd**2 - vm**3)/np.maximum(vsd**3,1e-12), 0.0)
    b3 = lambda a: [a[0::3], a[1::3], a[2::3]]
    feats = []
    feats += b3(n_t)                                   # trade counts per bucket (3)
    feats.append(ia_mu)                                # mean inter-arrival
    feats.append(ia_sd)                                # std inter-arrival
    feats.append(ia_sd/np.maximum(ia_mu,1e-9))         # burstiness (CV)
    feats.append(np.log1p(ia_max))                     # max gap
    feats.append(kl_num/np.maximum(kl_den,1e-12))      # Kyle lambda
    feats.append(pi_num/np.maximum(pi_den,1e-9))       # price impact per sqrt-vol
    feats.append(n_chg/nfull)                          # frac trades that move price
    feats.append(vm); feats.append(vsd)                # size mean/std
    feats.append(np.clip(vsk,-10,10))                  # size skew (clipped)
    feats.append(vmax/np.maximum(v1,1e-9))             # max trade share of volume
    feats.append(n_l10/nfull)                          # last-10s trade share
    feats.append(v_l10/np.maximum(v1,1e-9))            # last-10s volume share
    vwap = pv_sum/np.maximum(v1,1e-9)
    feats.append(vwap - 1.0)                           # vwap dev from normalized center
    X = np.stack([np.asarray(f_, f64) for f_ in feats], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(f'/kaggle/working/X25_{split}.npy', X)
    print(split, 'X25 done', X.shape, round(time.time()-t0), 's', flush=True)

build('train'); build('test')
