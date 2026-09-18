# X24c: X24 with empty-level (price==0) masking + liquidity-void features (NO cross-sectional, NO month-relative).
# Families: log-volume moments, signed-flow moments, trade-size asymmetry, spread/imb volatility,
# microprice extremes, order-cancel dynamics, early/late acceleration. Streaming bincount pattern (X22 style).
# Runs on comp raw feathers mounted in-kernel. Output: X24c_train.npy / X24c_test.npy (~48 cols). Dead rows excluded from price stats; dead incidence kept as features.
import os, time, gc
import numpy as np

DATA = os.environ.get('COMP', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = {'train': 1257637, 'test': 647896}
f64 = np.float64
# buckets: 0=[0,60) early, 1=[60,300) mid, 2=[300,600) late
edges = np.array([60.0, 300.0])

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
    # market accumulators per (sample, bucket)
    n_m = np.zeros(N3, f64); spread = np.zeros(N3, f64); spread2 = np.zeros(N3, f64)
    n_da = np.zeros(N3, f64); n_db = np.zeros(N3, f64); n_all = np.zeros(N3, f64)
    imb = np.zeros(N3, f64); imb2 = np.zeros(N3, f64)
    micro = np.zeros(N3, f64); microabs = np.zeros(N3, f64)
    slopea2 = np.zeros(N3, f64)
    dmid_sum = np.zeros(N3, f64); dmid_abs = np.zeros(N3, f64); dmid_sign = np.zeros(N3, f64)
    prev_mid = np.zeros(ns, f64); seen = np.zeros(ns, bool)
    for sid, sbp, a1, b1, av1, bv1, a2, b2 in ziter(f'{DATA}/{split}/market.feather',
            ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1','ask_price_2','bid_price_2']):
        bkt = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*3 + bkt
        a1f=a1.astype(f64); b1f=b1.astype(f64); av=av1.astype(f64); bv=bv1.astype(f64)
        alive = (a1f>0)&(b1f>0)  # price==0 marks an EMPTY level, not a price
        wgt = alive.astype(f64)
        mid=(a1f+b1f)/2; m=np.maximum(mid,1e-9)
        sp=(a1f-b1f)/m; im=(bv-av)/np.maximum(av+bv,1e-9)
        mic=(a1f*bv+b1f*av)/np.maximum(av+bv,1e-9)/m - 1
        n_all += np.bincount(lin, minlength=N3)
        n_m += np.bincount(lin, weights=wgt, minlength=N3)
        n_da += np.bincount(lin, weights=(a1f==0), minlength=N3)
        n_db += np.bincount(lin, weights=(b1f==0), minlength=N3)
        spread += np.bincount(lin, weights=np.where(alive, sp, 0.0), minlength=N3)
        spread2 += np.bincount(lin, weights=np.where(alive, sp*sp, 0.0), minlength=N3)
        imb += np.bincount(lin, weights=np.where(alive, im, 0.0), minlength=N3)
        imb2 += np.bincount(lin, weights=np.where(alive, im*im, 0.0), minlength=N3)
        micro += np.bincount(lin, weights=np.where(alive, mic, 0.0), minlength=N3)
        microabs += np.bincount(lin, weights=np.where(alive, np.abs(mic), 0.0), minlength=N3)
        slopea2 += np.bincount(lin, weights=np.where(alive&(a2.astype(f64)>0), (a2.astype(f64)-a1f)/m, 0.0), minlength=N3)
        # mid diffs within sample (alive rows only)
        pm = prev_mid[sid]
        has = seen[sid] & alive
        dm = np.where(has, np.log(m/np.maximum(pm,1e-12)), 0.0)
        dmid_sum += np.bincount(lin, weights=dm, minlength=N3)
        dmid_abs += np.bincount(lin, weights=np.abs(dm), minlength=N3)
        dmid_sign += np.bincount(lin, weights=np.sign(dm), minlength=N3)
        upd = alive
        prev_mid[sid] = np.where(upd, mid, prev_mid[sid]); seen[sid] = seen[sid] | alive
    print(split, 'market pass', round(time.time()-t0), 's', flush=True)
    n_m = np.maximum(n_m, 1.0)

    # transaction accumulators
    t_n = np.zeros(N3, f64); tv = np.zeros(N3, f64); tlv = np.zeros(N3, f64); tlv2 = np.zeros(N3, f64)
    tsv1 = np.zeros(N3, f64); tsv2 = np.zeros(N3, f64)  # signed vol, signed vol^2-moment
    for sid, sbp, pr, v, side in ziter(f'{DATA}/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side']):
        bkt = np.searchsorted(edges, sbp.astype(f64))
        lin = sid.astype(np.int64)*3 + bkt
        vf = v.astype(f64); sgn = np.where(side==0, 1.0, -1.0)
        lv = np.log1p(vf)
        t_n += np.bincount(lin, minlength=N3)
        tv += np.bincount(lin, weights=vf, minlength=N3)
        tlv += np.bincount(lin, weights=lv, minlength=N3)
        tlv2 += np.bincount(lin, weights=lv*lv, minlength=N3)
        tsv1 += np.bincount(lin, weights=vf*sgn, minlength=N3)
        tsv2 += np.bincount(lin, weights=vf*vf*sgn, minlength=N3)
    print(split, 'tx pass', round(time.time()-t0), 's', flush=True)
    t_nn = np.maximum(t_n, 1.0)

    # order accumulators (full window, per sample) - schema from features22.py
    o_new_b = np.zeros(ns, f64); o_new_s = np.zeros(ns, f64)
    o_can_b = np.zeros(ns, f64); o_can_s = np.zeros(ns, f64); o_n = np.zeros(ns, f64)
    for sid, sbp, pr, v, side, act in ziter(f'{DATA}/{split}/order.feather',
            ['sample_id','seconds_before_predict','price','volume','side','order_action']):
        vf = v.astype(f64); is_new = (act==0); buy = (side==0); s64 = sid.astype(np.int64)
        o_n += np.bincount(s64, minlength=ns)
        o_new_b += np.bincount(s64, weights=np.where(is_new&buy, vf, 0.0), minlength=ns)
        o_new_s += np.bincount(s64, weights=np.where(is_new&~buy, vf, 0.0), minlength=ns)
        o_can_b += np.bincount(s64, weights=np.where(~is_new&buy, vf, 0.0), minlength=ns)
        o_can_s += np.bincount(s64, weights=np.where(~is_new&~buy, vf, 0.0), minlength=ns)
    print(split, 'order pass', round(time.time()-t0), 's', flush=True)

    # assemble (all absolute, per-sample)
    def b3(x):  # split (ns*3) into 3 cols
        return [x[0::3]/1.0, x[1::3]/1.0, x[2::3]/1.0]
    feats = []
    nm = n_m
    sp_mu = spread/nm; sp_sd = np.sqrt(np.maximum(spread2/nm - sp_mu**2, 0))
    im_mu = imb/nm; im_sd = np.sqrt(np.maximum(imb2/nm - im_mu**2, 0))
    feats += b3(sp_sd)           # spread volatility per bucket (3)
    feats += b3(im_sd)           # imb volatility per bucket (3)
    feats += b3(microabs/nm)     # mean |microprice dev| per bucket (3)
    feats += b3(slopea2/nm)      # ask slope per bucket (3)
    feats += b3(dmid_abs/nm)     # realized |dlogmid| per bucket (3)
    feats += b3(dmid_sign/nm)    # sign consistency per bucket (3)
    lv_mu = tlv/t_nn; lv_sd = np.sqrt(np.maximum(tlv2/t_nn - lv_mu**2, 0))
    feats += b3(lv_mu); feats += b3(lv_sd)      # log-vol moments (6)
    feats += b3(tsv1/np.maximum(tv,1e-9))       # signed vol share (3)
    feats += b3(tsv2/np.maximum(tv*tv,1e-9))    # signed vol^2 moment (3)
    feats += b3(t_n)                            # trade counts (3)
    # accelerations/deltas
    e, m_, l = lambda arr: arr[0::3], lambda arr: arr[1::3], lambda arr: arr[2::3]
    feats.append(np.log1p(np.maximum(e(t_n),0)) - np.log1p(np.maximum(l(t_n),0)))  # trade accel
    feats.append(e(imb/nm) - l(imb/nm))                                          # imb drift
    feats.append(e(spread/nm) - l(spread/nm))                                    # spread drift
    feats.append(e(tsv1/np.maximum(tv,1e-9)) - l(tsv1/np.maximum(tv,1e-9)))# signed flow drift
    # liquidity-void (empty level) features
    na = np.maximum(n_all, 1.0)
    feats += b3(n_da/na)                            # frac bars with empty ask1 (3)
    feats += b3(n_db/na)                            # frac bars with empty bid1 (3)
    feats.append(e(n_da/na) - l(n_da/na))           # void drift ask
    feats.append(e(n_da/na) - e(n_db/na))           # void asymmetry (early)
    # order dynamics
    on = np.maximum(o_n, 1.0)
    feats.append((o_can_b + o_can_s) / np.maximum(o_new_b + o_new_s, 1e-9))      # cancel/new
    feats.append((o_new_b - o_new_s) / np.maximum(o_new_b + o_new_s, 1e-9))      # new imbalance
    feats.append((o_can_b - o_can_s) / np.maximum(o_can_b + o_can_s, 1e-9))      # cancel imbalance
    feats.append(np.log1p(on))                                                   # order count
    X = np.stack([np.asarray(f_, f64) for f_ in feats], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(f'/kaggle/working/X24c_{split}.npy', X)
    print(split, 'X24c done', X.shape, round(time.time()-t0), 's', flush=True)

build('train'); build('test')
print('DONE', flush=True)
