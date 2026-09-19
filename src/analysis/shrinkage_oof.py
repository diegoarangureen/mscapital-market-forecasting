# Offline shrinkage calibration on kfold455s5 per-model OOF preds.
# Idea: rows where the 25 models disagree are less certain; shrinking their norm reduces
# their weight in the cosine (cos weights rows by ||p||). Grid-search shrinkage factors.
# Usage: python3 shrinkage_oof.py <oof_models.npy> <full_y.npy> <full_month.npy>
import sys, json
import numpy as np

om_path, y_path, m_path = sys.argv[1], sys.argv[2], sys.argv[3]
OM = np.load(sys.argv[1], mmap_mode='r')   # (25, n_rows) f32, NaN outside each model's fold
y = np.load(y_path).astype(np.float64)
month = np.load(m_path)

def cos_np(p, t):
    p = p - p.mean(); t = t - t.mean()
    return float((p*t).sum() / (np.linalg.norm(p)+1e-12) / (np.linalg.norm(t)+1e-12))

# rows covered by >=2 models (folds 40-70); evaluate on 61-70 and 66-70
n_mod = np.sum(~np.isnan(OM), axis=0)
covered = n_mod >= 2
print('covered rows:', covered.sum())

mu = np.full(len(y), np.nan); sd = np.full(len(y), np.nan)
for i in range(0, len(y), 200000):
    blk = np.asarray(OM[:, i:i+200000], dtype=np.float64)
    mu[i:i+200000] = np.nanmean(blk, axis=0)
    sd[i:i+200000] = np.nanstd(blk, axis=0)

results = {}
for name, mask in [('61-70', (month>=61)&(month<=70)&covered), ('66-70', (month>=66)&covered)]:
    base = cos_np(mu[mask], y[mask])
    row = {'base': base}
    s = sd[mask]; m = mu[mask]; t = y[mask]
    sn = s / (np.abs(m).mean() + 1e-12)
    for lam in [0.25, 0.5, 1.0, 2.0, 4.0]:
        w = 1.0 / (1.0 + lam * sn / (sn.mean() + 1e-12))
        row[f'lam{lam}'] = cos_np(m * w, t)
    # also rank-based: shrink bottom q% by disagreement
    for q in [0.1, 0.25, 0.5]:
        thr = np.quantile(sn, 1-q)
        w = np.where(sn > thr, 0.5, 1.0)
        row[f'top{q}shrunk'] = cos_np(m * w, t)
    results[name] = row
    print(name, json.dumps(row, indent=1))
json.dump(results, open('/tmp/work/shrinkage_results.json','w'), indent=1)
