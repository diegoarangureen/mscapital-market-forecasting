# X32CTX: audited context pack - market-stream state over [60,600) plus drift into the
# last 60s. Diego's candidate 1 (WhatsApp 28-sep): OFI normalizado, persistencia del
# desequilibrio y profundidad en [60,600), y cambio hacia los ultimos 60 segundos.
# Absolute per-sample features only (XS trap confirmed 3x). Same ziter IO as build_x31.
# The market stream spans 600s (order/transaction span only 60s); X30/X31 flow features
# use <60s, so [60,600) is untouched context. No feature here repeats an X30/X31 column.
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ['MALLOC_ARENA_MAX'] = '1'
import numpy as np
BASE = os.environ.get('BASE', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = int(os.environ.get('NS', '1257637'))
SPLIT = os.environ.get('SPLIT', 'train')
OUT = os.environ.get('OUT', '/kaggle/working')

from audit.flows import ziter, chronological_order, ofi_level
from audit.io import save_feature_pack
os.makedirs(OUT, exist_ok=True)

f64 = np.float64
EPS = 1e-12

def build(stream_iter, NS):
    """Core: one market-stream pass -> (matrix, names). Pure numpy, testable off-Kaggle."""
    # row-level accumulators (context window = sbp in [60,600], recent = sbp < 60)
    n_ctx = np.zeros(NS, f64); n_rec = np.zeros(NS, f64)
    qi1_s_ctx = np.zeros(NS, f64); qi2_s_ctx = np.zeros(NS, f64)
    qi1_s_rec = np.zeros(NS, f64)
    qi1_pos = np.zeros(NS, f64)   # rows with qi1 > 0 (context, valid rows)
    qi1_neg = np.zeros(NS, f64)
    d1_s_ctx = np.zeros(NS, f64); d2_s_ctx = np.zeros(NS, f64)
    d1_s_rec = np.zeros(NS, f64)
    spread_s_ctx = np.zeros(NS, f64)
    # event-level (OFI transitions, time = later row)
    ofi1_ctx = np.zeros(NS, f64); ofi2_ctx = np.zeros(NS, f64)
    ofi1_rec = np.zeros(NS, f64)
    # mid at context boundaries: chronological first valid mid (~600s) and last mid with t>=60
    mid_first = np.zeros(NS, f64); mid_at60 = np.zeros(NS, f64)
    seen_first = np.zeros(NS, bool)
    for sid, sbp, a1, b1, av1, bv1, a2, b2, av2, bv2 in stream_iter:
        s64 = sid.astype(np.int64); t = sbp.astype(f64)
        a1f=a1.astype(f64); b1f=b1.astype(f64); av1f=av1.astype(f64); bv1f=bv1.astype(f64)
        a2f=a2.astype(f64); b2f=b2.astype(f64); av2f=av2.astype(f64); bv2f=bv2.astype(f64)
        valid = (a1f > 0) & (b1f > 0)
        ctx = (t >= 60.0) & valid
        rec = (t < 60.0) & valid
        qi1 = (bv1f-av1f)/np.maximum(bv1f+av1f, EPS)
        qi2 = (bv2f-av2f)/np.maximum(bv2f+av2f, EPS)
        mid = (a1f+b1f)/2.0
        np.add.at(n_ctx, s64[ctx], 1.0); np.add.at(n_rec, s64[rec], 1.0)
        np.add.at(qi1_s_ctx, s64[ctx], qi1[ctx]); np.add.at(qi2_s_ctx, s64[ctx], qi2[ctx])
        np.add.at(qi1_s_rec, s64[rec], qi1[rec])
        np.add.at(qi1_pos, s64[ctx], (qi1[ctx] > 0).astype(f64))
        np.add.at(qi1_neg, s64[ctx], (qi1[ctx] < 0).astype(f64))
        np.add.at(d1_s_ctx, s64[ctx], bv1f[ctx]+av1f[ctx]); np.add.at(d2_s_ctx, s64[ctx], bv2f[ctx]+av2f[ctx])
        np.add.at(d1_s_rec, s64[rec], bv1f[rec]+av1f[rec])
        np.add.at(spread_s_ctx, s64[ctx], (a1f[ctx]-b1f[ctx])/np.maximum(mid[ctx], EPS))
        # boundary mids (stream arrives grouped by sample but not time-sorted)
        order = chronological_order(s64, t)
        so=s64[order]; to=t[order]; mo=mid[order]; vo=valid[order]
        first_ok = vo & ~seen_first[so]
        idx = np.flatnonzero(first_ok)
        mid_first[so[idx]] = mo[idx]; seen_first[so[idx]] = True
        last60 = vo & (to >= 60.0)
        idx2 = np.flatnonzero(last60)
        # chronological pass: later assignment wins -> last row with t>=60 per sample
        mid_at60[so[idx2]] = mo[idx2]
        # OFI events
        b1o=b1f[order]; bv1o=bv1f[order]; a1o=a1f[order]; av1o=av1f[order]
        b2o=b2f[order]; bv2o=bv2f[order]; a2o=a2f[order]; av2o=av2f[order]
        same = so[1:] == so[:-1]
        e1 = ofi_level(b1o, bv1o, a1o, av1o, same)
        e2 = ofi_level(b2o, bv2o, a2o, av2o, same)
        tev = to[1:]; sev = so[1:]
        ectx = tev >= 60.0; erec = tev < 60.0
        np.add.at(ofi1_ctx, sev[ectx], e1[ectx]); np.add.at(ofi2_ctx, sev[ectx], e2[ectx])
        np.add.at(ofi1_rec, sev[erec], e1[erec])
    n_ctx[n_ctx == 0] = 1.0; n_rec[n_rec == 0] = 1.0
    mean_d1 = d1_s_ctx/n_ctx + 1.0
    qi1_m = qi1_s_ctx/n_ctx
    dom = np.maximum(qi1_pos, qi1_neg)  # rows agreeing with the dominant sign
    names = []; cols = []
    def add(name, x):
        names.append(name); cols.append(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    add('x32c_ofi1_nrm', np.clip(ofi1_ctx/mean_d1, -1e4, 1e4))
    add('x32c_ofi2_nrm', np.clip(ofi2_ctx/mean_d1, -1e4, 1e4))
    add('x32c_qi1_mean', qi1_m)
    add('x32c_qi2_mean', qi2_s_ctx/n_ctx)
    add('x32c_qi1_persist', dom/n_ctx)
    add('x32c_depth1_log', np.clip(np.log(d1_s_ctx/n_ctx + 1.0), 0, 30))
    add('x32c_depth2_log', np.clip(np.log(d2_s_ctx/n_ctx + 1.0), 0, 30))
    add('x32c_qi1_drift', qi1_s_rec/n_rec - qi1_m)
    add('x32c_ofi1_shift', np.clip(ofi1_rec/np.maximum(d1_s_rec/n_rec + 1.0, EPS)/60.0
                                   - ofi1_ctx/mean_d1/540.0, -1e3, 1e3))
    add('x32c_depth_drift', np.clip(np.log(d1_s_rec/n_rec + 1.0) - np.log(d1_s_ctx/n_ctx + 1.0), -10, 10))
    add('x32c_spread_ctx', np.clip(spread_s_ctx/n_ctx, 0, 1))
    add('x32c_mid_ret_ctx', np.clip(np.log(np.maximum(mid_at60, EPS)/np.maximum(mid_first, EPS)), -1, 1))
    return np.column_stack(cols), names

if __name__ == '__main__':
    t0 = time.time()
    it = ziter(f'{BASE}/{SPLIT}/market.feather',
               ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
                'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2'])
    X, names = build(it, NS)
    save_feature_pack(OUT, 'X32CTX', SPLIT, X, names)
    print('X32CTX saved', X.shape, 'features', len(names), round(time.time()-t0), 's', flush=True)
    print('DONE', flush=True)
