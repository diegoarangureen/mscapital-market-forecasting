"""CPU-only post-processing audit of the flow31 / base confirmation predictions (X33 step 1+3).
Inputs: per-job predictions.npz from kernel mscapital-confirm-tpu v1 + prepared-v2 y/month/train_nodata.
Four PREFIXED variants (declared before looking): raw / clip only / zeros-for-no-data only / both (= current).
No threshold search. These outer blocks were already consulted for flow31 vs base, so any
post-processing choice made from them carries selection risk; this report only explains the gap."""
import sys, json, glob, os
import numpy as np
sys.path.insert(0, '/tmp/repo/src/kaggle')
from audit.protocol import jobs
from audit.metrics import PredictionMean, cosine, postprocess

P = '/tmp/prepared/'; C = '/tmp/confirm/jobs/'
y = np.load(P + 'y.npy').astype(np.float64); month = np.load(P + 'month.npy'); nodata = np.load(P + 'train_nodata.npy').astype(bool)
cfg = json.load(open('/tmp/confirm/run_worker_00.json'))['config']
q = cfg['clip_quantiles']
n = len(y)
out = {'n_train_rows': n, 'nodata_rows_total': int(nodata.sum()), 'clip_quantiles': q}
exp = jobs(cfg)

def build(arm):
    res = {}
    for origin in (59, 64):
        es, sc = PredictionMean(n), PredictionMean(n)
        seed = {s: PredictionMean(n) for s in cfg['seeds']}
        for j in [j for j in exp if j.arm == arm and j.origin == origin]:
            p = np.load(C + j.key + '/predictions.npz')
            es.add(j.key, p['es_rows'], p['es_pred']); sc.add(j.key, p['score_rows'], p['score_pred'])
            seed[j.seed].add(j.key, p['score_rows'], p['score_pred'])
        seen = es.count > 0
        bounds = np.quantile(es.mean()[seen], q)
        rows = np.flatnonzero(sc.count > 0)
        res[origin] = dict(rows=rows, raw=sc.mean()[rows], bounds=bounds, es_n=int(seen.sum()),
                           es_count_hist=np.bincount(es.count[seen]).tolist(), sc_count=int(sc.count[rows].max()),
                           seed={s: m.mean()[rows] for s, m in seed.items()}, es_vals=es.mean()[seen])
    return res

def variants(raw, bounds, nd):
    return {'V0_raw': raw, 'V1_clip': postprocess(raw, bounds, None), 'V2_zeros': postprocess(raw, None, nd),
            'V3_clip+zeros(current)': postprocess(raw, bounds, nd)}

report = {}
for arm in ('base', 'flow31'):
    b = build(arm); report[arm] = {}
    allr = {k: np.full(n, np.nan) for k in ('V0_raw', 'V1_clip', 'V2_zeros', 'V3_clip+zeros(current)')}
    per_origin = {}; per_seed = {}
    for o, d in b.items():
        r = d['rows']; v = variants(d['raw'], d['bounds'], nodata[r])
        for k, x in v.items(): allr[k][r] = x
        per_origin[o] = {k: cosine(x, y[r]) for k, x in v.items()}
        per_seed[o] = {s: {k: cosine(x, y[r]) for k, x in variants(sv, d['bounds'], nodata[r]).items()} for s, sv in d['seed'].items()}
        raw = d['raw']
        report[arm][f'origin{o}_diag'] = dict(
            bounds=[float(d['bounds'][0]), float(d['bounds'][1])], es_rows=d['es_n'], es_models_per_row_hist=d['es_count_hist'],
            score_models_per_row=d['sc_count'], score_rows=int(len(r)),
            frac_clipped_low=float((raw < d['bounds'][0]).mean()), frac_clipped_high=float((raw > d['bounds'][1]).mean()),
            raw_std=float(raw.std()), es_std=float(d['es_vals'].std()),
            raw_q001=float(np.quantile(raw, q[0])), raw_q999=float(np.quantile(raw, q[1])),
            nodata_in_score_rows=int(nodata[r].sum()),
            nodata_cos_contrib_note='sum(y^2) share of nodata rows: %.3e' % float((y[r][nodata[r]] ** 2).sum() / (y[r] ** 2).sum()))
    rows = np.flatnonzero(np.isfinite(allr['V0_raw']))
    report[arm]['overall'] = {k: cosine(x[rows], y[rows]) for k, x in allr.items()}
    report[arm]['overall_ex66'] = {k: cosine(x[rows][month[rows] != 66], y[rows][month[rows] != 66]) for k, x in allr.items()}
    report[arm]['by_origin'] = per_origin
    report[arm]['by_seed'] = per_seed
    report[arm]['by_month'] = {int(mm): {k: cosine(x[rows][month[rows] == mm], y[rows][month[rows] == mm]) for k, x in allr.items()} for mm in np.unique(month[rows])}
    report[arm]['_vec'] = {k: x for k, x in allr.items()}
    report[arm]['_rows'] = rows

# paired deltas flow31 - base per variant (same rows)
rows = report['base']['_rows']; assert np.array_equal(rows, report['flow31']['_rows'])
paired = {}
for k in report['base']['_vec']:
    xb, xf = report['base']['_vec'][k][rows], report['flow31']['_vec'][k][rows]
    paired[k] = {'overall': cosine(xf, y[rows]) - cosine(xb, y[rows]),
                 'by_month': {int(mm): cosine(xf[month[rows] == mm], y[rows][month[rows] == mm]) - cosine(xb[month[rows] == mm], y[rows][month[rows] == mm]) for mm in np.unique(month[rows])}}
out['paired_flow31_minus_base'] = paired

# step 3: tail contributions on the processed-free raw ensemble, flow31
x = report['flow31']['_vec']['V0_raw'][rows]; yy = y[rows]
def shares(v, w, qs=(0.9, 0.99, 0.999)):
    a = np.abs(v); res = {}
    for qq in qs:
        t = np.quantile(a, qq); m = a >= t
        res[f'top{(1-qq)*100:g}%'] = float((w[m]).sum() / w.sum())
    return res
tail = {'sum_y2_share_by_|y|_tail': shares(yy, yy ** 2), 'sum_p2_share_by_|p|_tail': shares(x, x ** 2),
        'sum_py_share_by_|y|_tail': shares(yy, x * yy), 'sum_py_share_by_|p|_tail': shares(x, x * yy),
        'sum_p2_share_by_|y|_tail': shares(yy, x ** 2), 'sum_y2_share_by_|p|_tail': shares(x, yy ** 2)}
# contribution of |y| tail to cos numerator vs rest, and cosine on bulk only
a = np.abs(yy); t = np.quantile(a, 0.99); bulk = a < t
tail['cos_all'] = cosine(x, yy); tail['cos_ex_top1%_|y|'] = cosine(x[bulk], yy[bulk]); tail['sum_py_top1%_|y|_signed'] = float((x * yy)[~bulk].sum() / (x * yy).sum())
tail['frac_py_negative_in_top1%_|y|'] = float((x * yy)[~bulk].__lt__(0).mean())
tail['cos_with_y_winsor_q0.5/99.5(diagnostic only)'] = cosine(x, np.clip(yy, *np.quantile(yy, [0.005, 0.995])))
out['tail_contribution_flow31_raw'] = tail
# serialisable
for arm in ('base', 'flow31'):
    report[arm].pop('_vec'); report[arm].pop('_rows')
out['arms'] = report
json.dump(out, open('/tmp/repo/research/x33/postproc_audit.json', 'w'), indent=1, default=float)
print(json.dumps({k: out['arms'][k]['overall'] for k in ('base', 'flow31')}, indent=1))
print('ex66', json.dumps({k: out['arms'][k]['overall_ex66'] for k in ('base', 'flow31')}, indent=1))
print('paired', {k: round(v['overall'], 5) for k, v in paired.items()})
print(json.dumps(out['arms']['flow31']['origin59_diag'], indent=1)); print(json.dumps(out['arms']['flow31']['origin64_diag'], indent=1))
print(json.dumps(tail, indent=1))
