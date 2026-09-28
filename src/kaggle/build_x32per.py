# X32PER: audited persistence/response pack. Diego's candidate 2 (WhatsApp 28-sep):
# rachas de trades por signo y volumen, oleadas de cancelaciones por lado, reposicion
# de profundidad y respuesta del precio tras episodios de flujo. Hypothesis: current
# aggregates can hide distinct sequences with equal net volume.
# Absolute per-sample features only. Same ziter IO as build_x31. Streams span 60s.
# No feature repeats X30 (tsign_ac1 is count-autocorr, not run-lengths) or X31
# (gap stats, vpin, kyle) - run-length/wave/replenish/response are all new quantities.
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ['MALLOC_ARENA_MAX'] = '1'
import numpy as np
BASE = os.environ.get('BASE', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = int(os.environ.get('NS', '1257637'))
SPLIT = os.environ.get('SPLIT', 'train')
OUT = os.environ.get('OUT', '/kaggle/working')

from audit.flows import ziter, chronological_order
from audit.io import save_feature_pack
os.makedirs(OUT, exist_ok=True)

f64 = np.float64
EPS = 1e-12
NB30 = 30  # 2s buckets over the 60s window

def corr_rows(x, y):
    """Per-row Pearson between (NS,K) blocks; 0 where degenerate."""
    xm = x - x.mean(1, keepdims=True); ym = y - y.mean(1, keepdims=True)
    num = (xm*ym).sum(1)
    den = np.sqrt((xm*xm).sum(1)*(ym*ym).sum(1))
    return np.where(den > EPS, num/np.maximum(den, EPS), 0.0)

def build(tx_iter, od_iter, mk_iter, NS):
    # ---- transaction stream: run-lengths + signed volume buckets ----
    run_len_max = np.zeros(NS, f64); run_cnt = np.zeros(NS, f64)
    run_vol_max = np.zeros(NS, f64); tx_vol = np.zeros(NS, f64); tx_cnt = np.zeros(NS, f64)
    run_max_buy = np.zeros(NS, f64); run_max_sell = np.zeros(NS, f64)
    sv30 = np.zeros((NS, NB30), f64)
    for sid, sbp, pr, v, side in tx_iter:
        s64 = sid.astype(np.int64); t = sbp.astype(f64); vf = v.astype(f64)
        sgn = np.where(side == 0, 1.0, -1.0)
        np.add.at(tx_vol, s64, vf); np.add.at(tx_cnt, s64, 1.0)
        b = np.minimum((t // 2).astype(np.int64), NB30-1)
        np.add.at(sv30, (s64, b), sgn*vf)
        order = chronological_order(s64, t)
        so=s64[order]; go=sgn[order]; vo=vf[order]
        same = so[1:] == so[:-1]
        change = np.concatenate([[True], ~same | (go[1:] != go[:-1])])
        run_idx = np.cumsum(change) - 1
        starts = change
        run_sid = so[starts]; run_sign = go[starts]
        lens = np.bincount(run_idx).astype(f64)
        vols = np.bincount(run_idx, weights=go*vo)
        np.maximum.at(run_len_max, run_sid, lens)
        np.add.at(run_cnt, run_sid, 1.0)
        np.maximum.at(run_vol_max, run_sid, np.abs(vols))
        buy = run_sign > 0
        np.maximum.at(run_max_buy, run_sid[buy], lens[buy])
        np.maximum.at(run_max_sell, run_sid[~buy], lens[~buy])
    tx_cnt[tx_cnt == 0] = 1.0

    # ---- order stream: cancel waves + bucketed cancel/new volume per side ----
    can_bid = np.zeros(NS, f64); can_ask = np.zeros(NS, f64)
    wave_bid = np.zeros(NS, f64); wave_ask = np.zeros(NS, f64)
    c30_bid = np.zeros((NS, NB30), f64); c30_ask = np.zeros((NS, NB30), f64)
    n30_bid = np.zeros((NS, NB30), f64); n30_ask = np.zeros((NS, NB30), f64)
    for sid, sbp, v, side, act in od_iter:
        s64 = sid.astype(np.int64); t = sbp.astype(f64); vf = v.astype(f64)
        is_new = (act == 0); is_can = (act == 1); is_bid = (side == 0)
        b = np.minimum((t // 2).astype(np.int64), NB30-1)
        np.add.at(c30_bid, (s64, b), np.where(is_can & is_bid, vf, 0.0))
        np.add.at(c30_ask, (s64, b), np.where(is_can & ~is_bid, vf, 0.0))
        np.add.at(n30_bid, (s64, b), np.where(is_new & is_bid, vf, 0.0))
        np.add.at(n30_ask, (s64, b), np.where(is_new & ~is_bid, vf, 0.0))
        np.add.at(can_bid, s64, (is_can & is_bid).astype(f64))
        np.add.at(can_ask, s64, (is_can & ~is_bid).astype(f64))
        order = chronological_order(s64, t)
        so=s64[order]; to=t[order]; co=is_can[order]; bo=is_bid[order]
        same = so[1:] == so[:-1]
        pair = same & co[1:] & co[:-1] & (bo[1:] == bo[:-1])
        dt = np.abs(to[1:] - to[:-1])
        wave = pair & (dt <= 1.0)
        sw = so[1:]; bw = bo[1:]
        np.add.at(wave_bid, sw[wave & bw], 1.0)
        np.add.at(wave_ask, sw[wave & ~bw], 1.0)

    # ---- market stream: mid + qi1 per 2s bucket (<60s) ----
    mid_s = np.zeros((NS, NB30), f64); mid_n = np.zeros((NS, NB30), f64)
    qi1_s = np.zeros((NS, NB30), f64)
    for sid, sbp, a1, b1, av1, bv1 in mk_iter:
        s64 = sid.astype(np.int64); t = sbp.astype(f64)
        a1f=a1.astype(f64); b1f=b1.astype(f64); av1f=av1.astype(f64); bv1f=bv1.astype(f64)
        valid = (t < 60.0) & (a1f > 0) & (b1f > 0)
        s64=s64[valid]; t=t[valid]
        mid = (a1f[valid]+b1f[valid])/2.0
        qi1 = (bv1f[valid]-av1f[valid])/np.maximum(bv1f[valid]+av1f[valid], EPS)
        b = np.minimum((t // 2).astype(np.int64), NB30-1)
        np.add.at(mid_s, (s64, b), mid); np.add.at(mid_n, (s64, b), 1.0)
        np.add.at(qi1_s, (s64, b), qi1)

    # ---- assemble ----
    names = []; cols = []
    def add(name, x):
        names.append(name); cols.append(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    add('x32p_run_len_max', np.clip(np.log1p(run_len_max), 0, 20))
    add('x32p_run_len_mean', np.clip(np.where(run_cnt > 0, tx_cnt/np.maximum(run_cnt, 1.0), 0.0), 0, 1e3))
    add('x32p_run_vol_max_share', np.clip(run_vol_max/np.maximum(tx_vol, 1.0), 0, 1))
    add('x32p_run_imb', (run_max_buy-run_max_sell)/np.maximum(run_max_buy+run_max_sell, 1.0))
    add('x32p_cancel_wave_bid', wave_bid/np.maximum(can_bid, 1.0))
    add('x32p_cancel_wave_ask', wave_ask/np.maximum(can_ask, 1.0))
    # bucket-level replenishment: NEW volume in the bucket right after a cancel-active bucket
    rep_b = (np.where(c30_bid[:, :-1] > 0, n30_bid[:, 1:], 0.0)).sum(1)
    rep_a = (np.where(c30_ask[:, :-1] > 0, n30_ask[:, 1:], 0.0)).sum(1)
    add('x32p_replenish_bid', np.clip(rep_b/np.maximum(c30_bid.sum(1), 1.0), 0, 1e3))
    add('x32p_replenish_ask', np.clip(rep_a/np.maximum(c30_ask.sum(1), 1.0), 0, 1e3))
    # price response after flow episodes: corr of per-bucket flow vs NEXT-bucket mid return
    logmid = np.where(mid_n > 0, np.log(np.maximum(mid_s/np.maximum(mid_n, 1.0), EPS)), np.nan)
    # vectorized forward-fill along buckets, then backfill leading NaNs, then zero all-empty rows
    def ffill(a):
        idx = np.where(~np.isnan(a), np.arange(a.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        return np.take_along_axis(a, idx, axis=1)
    logmid = ffill(logmid)
    logmid = ffill(logmid[:, ::-1])[:, ::-1]
    logmid = np.where(np.isnan(logmid), 0.0, logmid)
    dret = logmid[:, 2:] - logmid[:, 1:-1]   # return of bucket b+1, b=0..27
    add('x32p_flow_resp', np.clip(corr_rows(sv30[:, :NB30-2], dret), -1, 1))
    qi1_m = qi1_s/np.maximum(mid_n, 1.0)
    add('x32p_qi_resp', np.clip(corr_rows(qi1_m[:, :NB30-2], dret), -1, 1))
    return np.column_stack(cols), names

if __name__ == '__main__':
    t0 = time.time()
    tx = ziter(f'{BASE}/{SPLIT}/transaction.feather',
               ['sample_id','seconds_before_predict','price','volume','side'])
    od = ziter(f'{BASE}/{SPLIT}/order.feather',
               ['sample_id','seconds_before_predict','volume','side','order_action'])
    mk = ziter(f'{BASE}/{SPLIT}/market.feather',
               ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1'])
    X, names = build(tx, od, mk, NS)
    save_feature_pack(OUT, 'X32PER', SPLIT, X, names)
    print('X32PER saved', X.shape, 'features', len(names), round(time.time()-t0), 's', flush=True)
    print('DONE', flush=True)
