# X30: Optiver-canon proprietary order-flow feature pack (train only; screen decides).
# Absolute per-sample features only (XS trap confirmed 3x). Built kernel-side from raw
# competition feather streams (pyarrow ziter + bincount accumulators; X29 IO pattern).
# Families: multi-level OFI (Cont-Kukanov-Stoikov), trade-sign autocorr, realized-vol
# estimators, order arrival/cancel intensity, queue-imbalance dynamics, microprice drift,
# book shape drift. Buckets b0=[0,10) b1=[10,30) b2=[30,60) (closest-to-predict, X21-style)
# plus [0,600) for market-stream features. order/transaction streams span only the last 60s.
# Encoding (verified from build_x2122_masked/build_x24 convention): side==0 -> buy/bid (+1), else -1; order_action 0=NEW, 1=CANCEL.
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ['MALLOC_ARENA_MAX'] = '1'
import numpy as np
BASE = os.environ.get('BASE', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = int(os.environ.get('NS', '1257637'))
SPLIT = os.environ.get('SPLIT', 'train')
OUT = os.environ.get('OUT', '/kaggle/working')

import pyarrow as pa
def ziter(path, cols, elems=1<<19):
    # files are a SINGLE giant record batch (probe v3): convert per-slice to cap peak RAM
    reader = pa.ipc.open_file(path)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        n = b.num_rows
        colrefs = [b.column(c) for c in cols]
        for s in range(0, n, elems):
            yield tuple(c[s:s+elems].to_numpy(zero_copy_only=False) for c in colrefs)

f64 = np.float64
edges = np.array([10.0, 30.0])   # bucket id via searchsorted: 0,1,2 (+3 for >=60 when present)
NB = 3
EPS = 1e-12

def bk(sbp):
    b = np.searchsorted(edges, sbp.astype(f64))
    return np.minimum(b, NB - 1)   # >=60 folds into b2 (order/tx only span 60s anyway)

t0 = time.time()
# ============ A) market stream: OFI L1/L2, RV family, QI dynamics, microprice, book shape ============
N = NS * NB
n_b = np.zeros(N, f64)
ofi1 = np.zeros(N, f64); ofi2 = np.zeros(N, f64)
rv = np.zeros(N, f64); rv_dn = np.zeros(N, f64); rv_up = np.zeros(N, f64)
qi1_s = np.zeros(N, f64); qi1_s2 = np.zeros(N, f64); qi2_s = np.zeros(N, f64)
spr_s = np.zeros(N, f64)
mic_s = np.zeros(N, f64); mid_s = np.zeros(N, f64); micmid_s = np.zeros(N, f64); spr_s2 = np.zeros(N, f64)
qi1_tw_s = np.zeros(N, f64); tw_w = np.zeros(N, f64)
bv1_s = np.zeros(NS, f64); bv2_s = np.zeros(NS, f64); av1_s = np.zeros(NS, f64); av2_s = np.zeros(NS, f64); n_all = np.zeros(NS, f64)
rv600 = np.zeros(NS, f64)
# per (sample, 10s sub-bucket) hi/lo of mid for Parkinson + sub-RV for vol-of-vol
NSB = 60  # 600s / 10s
mid_hi = np.full(NS * NSB, -np.inf); mid_lo = np.full(NS * NSB, np.inf)
sub_rv = np.zeros(NS * NSB, f64)

tvwap_s = np.zeros(NS, f64); tvol_s = np.zeros(NS, f64); tcount_s = np.zeros(NS, f64); vwap_mid_s = np.zeros(NS, f64)
for sid, sbp, a1, b1, av1, bv1, a2, b2, av2, bv2, txp, txv, txc in ziter(f'{BASE}/{SPLIT}/market.feather',
        ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
         'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2',
         'transaction_avgprice','transaction_volume','transaction_count']):
    s64 = sid.astype(np.int64); t = sbp.astype(f64)
    a1f=a1.astype(f64); b1f=b1.astype(f64); av1f=av1.astype(f64); bv1f=bv1.astype(f64)
    a2f=a2.astype(f64); b2f=b2.astype(f64); av2f=av2.astype(f64); bv2f=bv2.astype(f64)
    valid = (a1f > 0) & (b1f > 0)
    mid = np.where(valid, (a1f+b1f)/2.0, np.nan)
    micro = np.where(valid, (a1f*bv1f + b1f*av1f)/np.maximum(av1f+bv1f, EPS), np.nan)
    qi1 = np.where(valid, (bv1f-av1f)/np.maximum(bv1f+av1f, EPS), 0.0)
    qi2 = (bv2f-av2f)/np.maximum(bv2f+av2f, EPS)
    spr = np.where(valid, a1f-b1f, np.nan)
    # sort chunk by (sample, time) for diff-based features
    order = np.lexsort((t, s64))
    s64o=s64[order]; to=t[order]
    same = s64o[1:] == s64o[:-1]
    # OFI L1/L2 (Cont-Kukanov-Stoikov event rule)
    b1o=b1f[order]; bv1o=bv1f[order]; a1o=a1f[order]; av1o=av1f[order]
    b2o=b2f[order]; bv2o=bv2f[order]; a2o=a2f[order]; av2o=av2f[order]
    def ofi_level(pb, pv, pa, pav, same):
        db = pb[1:] - pb[:-1]; dvb = pv[1:] - pv[:-1]
        da = pa[1:] - pa[:-1]; dva = pav[1:] - pav[:-1]
        bid_c = np.where(db > 0, pv[1:], np.where(db < 0, -pv[:-1], dvb))
        ask_c = np.where(da < 0, pav[1:], np.where(da > 0, -pav[:-1], dva))
        e = (bid_c - ask_c) * same
        return e
    e1 = ofi_level(b1o, bv1o, a1o, av1o, same)
    e2 = ofi_level(b2o, bv2o, a2o, av2o, same)
    dmask = to[1:] < 60.0
    lin_d = s64o[1:][dmask] * NB + bk(to[1:][dmask])
    np.add.at(ofi1, lin_d, e1[dmask]); np.add.at(ofi2, lin_d, e2[dmask])
    # RV from mid log returns
    mido = mid[order]
    r = np.diff(np.log(np.where(mido > 0, mido, np.nan)))
    r = np.where(same & np.isfinite(r), r, 0.0)
    np.add.at(rv, lin_d, (r*r)[dmask])
    np.add.at(rv_dn, lin_d, np.where(r < 0, r*r, 0.0)[dmask]); np.add.at(rv_up, lin_d, np.where(r > 0, r*r, 0.0)[dmask])
    np.add.at(rv600, s64o[1:], r*r)
    sub = s64o[1:] * NSB + np.minimum((to[1:] // 10).astype(np.int64), NSB-1)
    np.add.at(sub_rv, sub, r*r)
    # per-bucket means - buckets are [0,10) [10,30) [30,60); market rows >=60s are EXCLUDED
    # from bucketed stats (bk() folds >=60 into b2 for the 60s-spanning order/tx streams only).
    bm = t < 60.0
    lin = s64[bm] * NB + bk(t[bm])
    w = np.isfinite(mid)
    wb = w[bm]
    qi1b = qi1[bm]; qi2b = qi2[bm]; microb = micro[bm]; midb = mid[bm]; sprb = spr[bm]; tb = t[bm]
    np.add.at(n_b, lin, wb)
    np.add.at(qi1_s, lin, np.where(wb, qi1b, 0.0)); np.add.at(qi1_s2, lin, np.where(wb, qi1b*qi1b, 0.0))
    np.add.at(qi2_s, lin, np.where(wb, qi2b, 0.0))
    np.add.at(mic_s, lin, np.where(wb, np.nan_to_num(microb), 0.0))
    np.add.at(mid_s, lin, np.where(wb, np.nan_to_num(midb), 0.0))
    np.add.at(micmid_s, lin, np.where(wb, np.nan_to_num((microb-midb)/np.maximum(midb, EPS)), 0.0))
    np.add.at(spr_s2, lin, np.where(wb, np.nan_to_num(sprb*sprb), 0.0))
    np.add.at(spr_s, lin, np.where(wb, np.nan_to_num(sprb), 0.0))
    wt = 1.0/(tb+1.0)
    np.add.at(qi1_tw_s, lin, qi1b*wt); np.add.at(tw_w, lin, wt)
    # sub-bucket mid hi/lo (valid rows only)
    wr = w & (mid > 0)
    sub_all = s64 * NSB + np.minimum((t // 10).astype(np.int64), NSB-1)
    np.minimum.at(mid_lo, sub_all[wr], mid[wr]); np.maximum.at(mid_hi, sub_all[wr], mid[wr])
    # depth sums (per sample)
    np.add.at(bv1_s, s64, bv1f); np.add.at(bv2_s, s64, bv2f)
    np.add.at(av1_s, s64, av1f); np.add.at(av2_s, s64, av2f); np.add.at(n_all, s64, 1.0)
    txvf = txv.astype(f64)
    np.add.at(tvol_s, s64, txvf); np.add.at(tcount_s, s64, txc.astype(f64))
    np.add.at(tvwap_s, s64, np.nan_to_num(txp.astype(f64))*txvf)
    np.add.at(vwap_mid_s, s64, np.where(w, txvf*np.nan_to_num(mid), 0.0))

n_b[n_b == 0] = 1.0
print('market stream done', round(time.time()-t0), 's', flush=True)

# ============ B) transaction stream: trade-sign autocorr + flow ============
ts_cnt = np.zeros(NS, f64); ts_ac1 = np.zeros(NS, f64); ts_ac2 = np.zeros(NS, f64); ts_ac3 = np.zeros(NS, f64)
ts_signvol = np.zeros(NS, f64); ts_vol = np.zeros(NS, f64)
ts_sv10 = np.zeros(NS, f64); ts_v10 = np.zeros(NS, f64)
for sid, sbp, pr, v, side in ziter(f'{BASE}/{SPLIT}/transaction.feather',
        ['sample_id','seconds_before_predict','price','volume','side']):
    s64 = sid.astype(np.int64); t = sbp.astype(f64); vf = v.astype(f64)
    sgn = np.where(side == 0, 1.0, -1.0)
    order = np.lexsort((t, s64))
    so=s64[order]; go=sgn[order]; to=t[order]
    same = so[1:] == so[:-1]
    np.add.at(ts_ac1, so[1:], go[1:]*go[:-1]*same)
    same2 = so[2:] == so[:-2]; np.add.at(ts_ac2, so[2:], go[2:]*go[:-2]*same2)
    same3 = so[3:] == so[:-3]; np.add.at(ts_ac3, so[3:], go[3:]*go[:-3]*same3)
    np.add.at(ts_cnt, s64, 1.0)
    np.add.at(ts_signvol, s64, sgn*vf); np.add.at(ts_vol, s64, vf)
    m10 = t < 10.0
    np.add.at(ts_sv10, s64[m10], (sgn*vf)[m10]); np.add.at(ts_v10, s64[m10], vf[m10])
ts_cnt[ts_cnt == 0] = 1.0
print('transaction stream done', round(time.time()-t0), 's', flush=True)

# ============ C) order stream: arrival/cancel intensity ============
o_add = np.zeros(NS, f64); o_can = np.zeros(NS, f64)
o_add_b = np.zeros(NS, f64); o_can_b = np.zeros(NS, f64); o_add_a = np.zeros(NS, f64); o_can_a = np.zeros(NS, f64)
for sid, sbp, pr, v, side, act in ziter(f'{BASE}/{SPLIT}/order.feather',
        ['sample_id','seconds_before_predict','price','volume','side','order_action']):
    s64 = sid.astype(np.int64)
    is_new = (act == 0); is_can = (act == 1)
    is_bid = (side == 0)
    np.add.at(o_add, s64, is_new); np.add.at(o_can, s64, is_can)
    np.add.at(o_add_b, s64, is_new & is_bid); np.add.at(o_can_b, s64, is_can & is_bid)
    np.add.at(o_add_a, s64, is_new & ~is_bid); np.add.at(o_can_a, s64, is_can & ~is_bid)
print('order stream done', round(time.time()-t0), 's', flush=True)

# ============ assemble features ============
def bucketed(arr, b):  # mean per bucket b
    return (arr.reshape(NS, NB)[:, b] / n_b.reshape(NS, NB)[:, b])

names = []; cols = []
def add(name, x):
    names.append(name); cols.append(x.astype(np.float32))

# OFI sums per bucket + total
for i in range(3):
    add(f'x30_ofi1_b{i}', ofi1.reshape(NS, NB)[:, i])
    add(f'x30_ofi2_b{i}', ofi2.reshape(NS, NB)[:, i])
add('x30_ofi1_all', ofi1.reshape(NS, NB).sum(1))
add('x30_ofi2_all', ofi2.reshape(NS, NB).sum(1))
# RV
rv_b = rv.reshape(NS, NB); rvdn_b = rv_dn.reshape(NS, NB); rvup_b = rv_up.reshape(NS, NB)
add('x30_rv_60', np.sqrt(rv_b.sum(1)))
add('x30_rv_600', np.sqrt(rv600))
add('x30_semi_ratio', (rvdn_b.sum(1)+EPS) / (rvup_b.sum(1)+EPS))
pk = (np.log(np.maximum(mid_hi, EPS)/np.maximum(mid_lo, EPS))**2) / (4*np.log(2))
pk = np.where(np.isfinite(pk) & (mid_hi > mid_lo), pk, 0.0).reshape(NS, NSB)
add('x30_parkinson', np.sqrt(pk[:, :6].sum(1)))          # last 60s
sub_rv2 = np.sqrt(np.maximum(sub_rv, 0.0)).reshape(NS, NSB)
add('x30_vol_of_vol', sub_rv2[:, :6].std(1))
# trade sign
add('x30_tsign_ac1', ts_ac1/ts_cnt); add('x30_tsign_ac2', ts_ac2/ts_cnt); add('x30_tsign_ac3', ts_ac3/ts_cnt)
add('x30_signvol_60', ts_signvol/np.maximum(ts_vol, EPS))
add('x30_flow_shift', ts_sv10/np.maximum(ts_v10, EPS) - ts_signvol/np.maximum(ts_vol, EPS))
# order intensity
add('x30_ord_add_rate', o_add/60.0)
add('x30_ord_cancel_ratio', o_can/np.maximum(o_add, 1.0))
add('x30_ord_net_bid', o_add_b - o_can_b)
add('x30_ord_net_ask', o_add_a - o_can_a)
# QI dynamics
qi1_b = qi1_s.reshape(NS, NB)/n_b.reshape(NS, NB)
qi2_b = qi2_s.reshape(NS, NB)/n_b.reshape(NS, NB)
qi1_var = qi1_s2.reshape(NS, NB)/n_b.reshape(NS, NB) - qi1_b**2
add('x30_qi1_drift', qi1_b[:, 0] - qi1_b[:, 2])
add('x30_qi1_vol', np.sqrt(np.maximum(qi1_var.mean(1), 0.0)))
add('x30_qi2_drift', qi2_b[:, 0] - qi2_b[:, 2])
add('x30_qi1_tw', qi1_tw_s.reshape(NS, NB).sum(1)/np.maximum(tw_w.reshape(NS, NB).sum(1), EPS))
# microprice dynamics
mic_b = mic_s.reshape(NS, NB)/n_b.reshape(NS, NB)
mid_b = mid_s.reshape(NS, NB)/n_b.reshape(NS, NB)
midbar = np.maximum(mid_b.mean(1), EPS)
add('x30_micro_slope', (mic_b[:, 0] - mic_b[:, 2])/midbar)
add('x30_mid_slope', (mid_b[:, 0] - mid_b[:, 2])/midbar)
add('x30_micro_mid_dev', (micmid_s.reshape(NS, NB)/n_b.reshape(NS, NB)).mean(1))
add('x30_spread_rms', np.sqrt(np.maximum((spr_s2.reshape(NS, NB)/n_b.reshape(NS, NB)).mean(1), 0.0))/midbar)
spr_b = spr_s.reshape(NS, NB)/n_b.reshape(NS, NB)
add('x30_spread_drift', (spr_b[:, 0] - spr_b[:, 2])/midbar)
# book shape
n_all[n_all == 0] = 1.0
add('x30_vwap_dev_600', (tvwap_s/np.maximum(tvol_s, EPS) - vwap_mid_s/np.maximum(tvol_s, EPS))/midbar)
add('x30_tvol_600_log', np.log1p(tvol_s))
add('x30_tcount_600_log', np.log1p(tcount_s))
add('x30_depth_ratio_bid', (bv2_s/n_all)/np.maximum(bv1_s/n_all, EPS))
add('x30_depth_ratio_ask', (av2_s/n_all)/np.maximum(av1_s/n_all, EPS))

X = np.column_stack(cols)
np.save(f'{OUT}/X30_{SPLIT}.npy', X)
np.save(f'{OUT}/X30_{SPLIT}_names.npy', np.array(names))
print('X30 saved', X.shape, 'features', len(names), round(time.time()-t0), 's', flush=True)
print('DONE', flush=True)
