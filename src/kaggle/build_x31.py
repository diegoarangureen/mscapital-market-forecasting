# X31: FLOW-family extension pack (train only; paired 3-arm screen decides).
# Motivation: X30 FLOW ablation carried ~60% of the pack signal (f1 +0.0040, f5 +0.0029);
# VOL flat. X31 adds NONLINEAR / distributional FLOW variants a RealMLP cannot synthesize
# from the existing linear per-bucket sums: sub-bucket burstiness, time-weighted OFI,
# volume-weighted order flow, inter-arrival gap stats, Kyle lambda, VPIN-lite.
# Absolute per-sample features only (XS trap confirmed 3x). Same ziter IO as build_x30.
# Streams: market spans 600s (bucketed stats use <60s), order/transaction span last 60s.
# Encoding: side==0 -> buy/bid (+1), else -1; order_action 0=NEW, 1=CANCEL.
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
    reader = pa.ipc.open_file(path)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        n = b.num_rows
        colrefs = [b.column(c) for c in cols]
        for s in range(0, n, elems):
            yield tuple(c[s:s+elems].to_numpy(zero_copy_only=False) for c in colrefs)

f64 = np.float64
edges = np.array([10.0, 30.0])
NB = 3
NSB6 = 6   # 6 x 10s sub-buckets over the 60s flow window
EPS = 1e-12

def bk(sbp):
    b = np.searchsorted(edges, sbp.astype(f64))
    return np.minimum(b, NB - 1)

t0 = time.time()
# ============ A) market stream: OFI sub-bucket burstiness, tw-OFI, per-bucket depth, qi2_tw ============
n_all = np.zeros(NS, f64)
bv1_s = np.zeros(NS, f64); av1_s = np.zeros(NS, f64); bv2_s = np.zeros(NS, f64); av2_s = np.zeros(NS, f64)
bv1_b = np.zeros(NS * NB, f64); av1_b = np.zeros(NS * NB, f64)
bv2_b = np.zeros(NS * NB, f64); av2_b = np.zeros(NS * NB, f64)
ofi1_sub = np.zeros(NS * NSB6, f64); ofi2_sub = np.zeros(NS * NSB6, f64)
ofi1_tw = np.zeros(NS, f64); ofi2_tw = np.zeros(NS, f64)
qi2_tw_s = np.zeros(NS, f64); qi2_tw_w = np.zeros(NS, f64)
for sid, sbp, a1, b1, av1, bv1, a2, b2, av2, bv2 in ziter(f'{BASE}/{SPLIT}/market.feather',
        ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
         'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2']):
    s64 = sid.astype(np.int64); t = sbp.astype(f64)
    a1f=a1.astype(f64); b1f=b1.astype(f64); av1f=av1.astype(f64); bv1f=bv1.astype(f64)
    a2f=a2.astype(f64); b2f=b2.astype(f64); av2f=av2.astype(f64); bv2f=bv2.astype(f64)
    valid = (a1f > 0) & (b1f > 0)
    qi2 = (bv2f-av2f)/np.maximum(bv2f+av2f, EPS)
    order = np.lexsort((t, s64))
    s64o=s64[order]; to=t[order]
    same = s64o[1:] == s64o[:-1]
    b1o=b1f[order]; bv1o=bv1f[order]; a1o=a1f[order]; av1o=av1f[order]
    b2o=b2f[order]; bv2o=bv2f[order]; a2o=a2f[order]; av2o=av2f[order]
    def ofi_level(pb, pv, pa, pav, same):
        db = pb[1:] - pb[:-1]; dvb = pv[1:] - pv[:-1]
        da = pa[1:] - pa[:-1]; dva = pav[1:] - pav[:-1]
        bid_c = np.where(db > 0, pv[1:], np.where(db < 0, -pv[:-1], dvb))
        ask_c = np.where(da < 0, pav[1:], np.where(da > 0, -pav[:-1], dva))
        return (bid_c - ask_c) * same
    e1 = ofi_level(b1o, bv1o, a1o, av1o, same)
    e2 = ofi_level(b2o, bv2o, a2o, av2o, same)
    dmask = to[1:] < 60.0
    sd = s64o[1:][dmask]; td = to[1:][dmask]
    sub6 = sd * NSB6 + np.minimum((td // 10).astype(np.int64), NSB6-1)
    np.add.at(ofi1_sub, sub6, e1[dmask]); np.add.at(ofi2_sub, sub6, e2[dmask])
    wt = 1.0/(td+1.0)
    np.add.at(ofi1_tw, sd, e1[dmask]*wt); np.add.at(ofi2_tw, sd, e2[dmask]*wt)
    # time-weighted qi2 over the 60s window
    m60 = t < 60.0
    wtv = 1.0/(t[m60]+1.0)
    np.add.at(qi2_tw_s, s64[m60], qi2[m60]*wtv); np.add.at(qi2_tw_w, s64[m60], wtv)
    # depth sums: per sample + per bucket (bucketed rows <60s only)
    np.add.at(bv1_s, s64, bv1f); np.add.at(av1_s, s64, av1f)
    np.add.at(bv2_s, s64, bv2f); np.add.at(av2_s, s64, av2f)
    np.add.at(n_all, s64, 1.0)
    lin = s64[m60] * NB + bk(t[m60])
    np.add.at(bv1_b, lin, bv1f[m60]); np.add.at(av1_b, lin, av1f[m60])
    np.add.at(bv2_b, lin, bv2f[m60]); np.add.at(av2_b, lin, av2f[m60])
n_all[n_all == 0] = 1.0
print('market stream done', round(time.time()-t0), 's', flush=True)

# ============ B) transaction stream: Kyle lambda, VPIN-lite, gap stats ============
ts_vol = np.zeros(NS, f64); ts_cnt = np.zeros(NS, f64)
sv_sub = np.zeros(NS * NSB6, f64); v_sub = np.zeros(NS * NSB6, f64)
kyle_num = np.zeros(NS, f64); kyle_den = np.zeros(NS, f64)
tg_sum = np.zeros(NS, f64); tg_sum2 = np.zeros(NS, f64); tg_cnt = np.zeros(NS, f64)
for sid, sbp, pr, v, side in ziter(f'{BASE}/{SPLIT}/transaction.feather',
        ['sample_id','seconds_before_predict','price','volume','side']):
    s64 = sid.astype(np.int64); t = sbp.astype(f64); vf = v.astype(f64)
    sgn = np.where(side == 0, 1.0, -1.0)
    order = np.lexsort((t, s64))
    so=s64[order]; go=sgn[order]; vo=vf[order]; to=t[order]; po=pr[order].astype(f64)
    same = so[1:] == so[:-1]
    # Kyle lambda from tx price log-returns vs signed volume
    r = np.diff(np.log(np.where(po > 0, po, np.nan)))
    r = np.where(same & np.isfinite(r), r, 0.0)
    sv = go[1:]*vo[1:]
    np.add.at(kyle_num, so[1:], r*sv); np.add.at(kyle_den, so[1:], sv*sv)
    # inter-arrival gaps
    gap = np.where(same, np.abs(to[1:]-to[:-1]), 0.0)
    np.add.at(tg_sum, so[1:], gap); np.add.at(tg_sum2, so[1:], gap*gap); np.add.at(tg_cnt, so[1:], same)
    np.add.at(ts_vol, s64, vf); np.add.at(ts_cnt, s64, 1.0)
    sub6 = s64 * NSB6 + np.minimum((t // 10).astype(np.int64), NSB6-1)
    np.add.at(sv_sub, sub6, sgn*vf); np.add.at(v_sub, sub6, vf)
ts_cnt[ts_cnt == 0] = 1.0
print('transaction stream done', round(time.time()-t0), 's', flush=True)

# ============ C) order stream: volume-weighted flow + NEW-order gaps ============
ov_new_b = np.zeros(NS, f64); ov_new_a = np.zeros(NS, f64)
ov_can_b = np.zeros(NS, f64); ov_can_a = np.zeros(NS, f64)
og_sum = np.zeros(NS, f64); og_sum2 = np.zeros(NS, f64); og_cnt = np.zeros(NS, f64)
for sid, sbp, v, side, act in ziter(f'{BASE}/{SPLIT}/order.feather',
        ['sample_id','seconds_before_predict','volume','side','order_action']):
    s64 = sid.astype(np.int64); t = sbp.astype(f64); vf = v.astype(f64)
    is_new = (act == 0); is_can = (act == 1); is_bid = (side == 0)
    np.add.at(ov_new_b, s64, np.where(is_new & is_bid, vf, 0.0))
    np.add.at(ov_new_a, s64, np.where(is_new & ~is_bid, vf, 0.0))
    np.add.at(ov_can_b, s64, np.where(is_can & is_bid, vf, 0.0))
    np.add.at(ov_can_a, s64, np.where(is_can & ~is_bid, vf, 0.0))
    order = np.lexsort((t, s64))
    so=s64[order]; to=t[order]; no=(is_new[order])
    same = so[1:] == so[:-1]
    both_new = same & no[1:] & no[:-1]
    gap = np.where(both_new, np.abs(to[1:]-to[:-1]), 0.0)
    np.add.at(og_sum, so[1:], gap); np.add.at(og_sum2, so[1:], gap*gap); np.add.at(og_cnt, so[1:], both_new)
print('order stream done', round(time.time()-t0), 's', flush=True)

# ============ assemble ============
names = []; cols = []
def add(name, x):
    names.append(name); cols.append(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))

mean_depth = (bv1_s + av1_s) / n_all + 1.0
o1s = ofi1_sub.reshape(NS, NSB6); o2s = ofi2_sub.reshape(NS, NSB6)
add('x31_ofi1_vol', np.clip(o1s.std(1)/mean_depth, 0, 1e4))
add('x31_ofi2_vol', np.clip(o2s.std(1)/mean_depth, 0, 1e4))
add('x31_ofi1_accel', np.clip((o1s[:, :2].mean(1) - o1s[:, 4:].mean(1))/mean_depth, -1e4, 1e4))
add('x31_ofi2_accel', np.clip((o2s[:, :2].mean(1) - o2s[:, 4:].mean(1))/mean_depth, -1e4, 1e4))
add('x31_ofi1_tw', np.clip(ofi1_tw/mean_depth, -1e4, 1e4))
add('x31_ofi2_tw', np.clip(ofi2_tw/mean_depth, -1e4, 1e4))
add('x31_qi2_tw', qi2_tw_s/np.maximum(qi2_tw_w, EPS))
add('x31_book_imb12', np.clip((bv1_s+bv2_s-av1_s-av2_s)/np.maximum(bv1_s+bv2_s+av1_s+av2_s, EPS), -1, 1))
bv1q = bv1_b.reshape(NS, NB); av1q = av1_b.reshape(NS, NB)
bv2q = bv2_b.reshape(NS, NB); av2q = av2_b.reshape(NS, NB)
drb = bv2q/np.maximum(bv1q, EPS); dra = av2q/np.maximum(av1q, EPS)
add('x31_dratio_bid_drift', np.clip(drb[:, 0]-drb[:, 2], -1e3, 1e3))
add('x31_dratio_ask_drift', np.clip(dra[:, 0]-dra[:, 2], -1e3, 1e3))
vol_scale = np.maximum(ts_vol, 1.0)
add('x31_ord_vnet_new', np.clip((ov_new_b-ov_new_a)/vol_scale, -1e4, 1e4))
add('x31_ord_vnet_can', np.clip((ov_can_b-ov_can_a)/vol_scale, -1e4, 1e4))
og_m = og_sum/np.maximum(og_cnt, 1.0)
og_v = og_sum2/np.maximum(og_cnt, 1.0) - og_m**2
add('x31_ord_gap_mean', np.clip(og_m, 0, 60))
add('x31_ord_gap_cv', np.clip(np.sqrt(np.maximum(og_v, 0.0))/np.maximum(og_m, EPS), 0, 1e3))
tg_m = tg_sum/np.maximum(tg_cnt, 1.0)
tg_v = tg_sum2/np.maximum(tg_cnt, 1.0) - tg_m**2
add('x31_tx_gap_cv', np.clip(np.sqrt(np.maximum(tg_v, 0.0))/np.maximum(tg_m, EPS), 0, 1e3))
mean_trade = ts_vol/ts_cnt
add('x31_kyle_lambda', np.clip(kyle_num/np.maximum(kyle_den, EPS)*mean_trade, -10, 10))
svs = sv_sub.reshape(NS, NSB6); vs = v_sub.reshape(NS, NSB6)
vpin_b = np.abs(svs)/np.maximum(vs, EPS)
add('x31_vpin_mean', vpin_b.mean(1))
add('x31_vpin_max', vpin_b.max(1))

X = np.column_stack(cols)
np.save(f'{OUT}/X31_{SPLIT}.npy', X)
np.save(f'{OUT}/X31_{SPLIT}_names.npy', np.array(names))
print('X31 saved', X.shape, 'features', len(names), round(time.time()-t0), 's', flush=True)
print('DONE', flush=True)
