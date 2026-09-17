# Build XS75 cross-sectional features (bestwater 689f recipe piece) on our 298f-advdrop base.
# 30 within-month pct-ranks + 15 within-month z-scores + 30 cross-feature ranks.
# Selection: LGBM gain top30 on 200k subsample of train.
# Train XS: within-train-month. Test XS: within-test-month (transductive, per bestwater).
# Output: /kaggle/working/XS_train.npy (n,75), XS_test.npy (m,75), xs_top30.npy
import os, time
import numpy as np

DATA = os.environ.get('DATA', '/kaggle/input/datasets/diegoaranguren/mscapital-matrices')
XTRA = os.environ.get('XTRA', '/kaggle/input/datasets/diegoaranguren/mscapital-x21')
XTRA2 = os.environ.get('XTRA2', '/kaggle/input/datasets/diegoaranguren/mscapital-x22')

t0 = time.time()
Xtr = np.load(f'{DATA}/full_train.npy')
X1 = np.load(f'{XTRA}/X21_train.npy').astype(np.float32)
X2 = np.load(f'{XTRA2}/X22_train.npy').astype(np.float32)
Xtr = np.concatenate([Xtr, np.nan_to_num(X1), np.nan_to_num(X2)], axis=1)
del X1, X2
Xte = np.load(f'{DATA}/full_test.npy')
X1 = np.load(f'{XTRA}/X21_test.npy').astype(np.float32)
X2 = np.load(f'{XTRA2}/X22_test.npy').astype(np.float32)
Xte = np.concatenate([Xte, np.nan_to_num(X1), np.nan_to_num(X2)], axis=1)
del X1, X2
DROP = [301, 267, 268, 299, 257]
Xtr = np.delete(Xtr, DROP, axis=1)
Xte = np.delete(Xte, DROP, axis=1)
y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32)
month_tr = np.load(f'{DATA}/full_month.npy')
print('base298', Xtr.shape, Xte.shape, flush=True)

# test month from 0726 test.csv (aligned by sample_id)
import pandas as pd
from pathlib import Path
d0726 = None
for cand in ['/kaggle/input/rfmf-0726data', '/kaggle/input/kernels/yunsuxiaozi/rfmf-0726data']:
    if Path(cand).exists() and (Path(cand)/'test.csv').exists():
        d0726 = cand; break
if d0726 is None:
    import subprocess
    r = subprocess.run(['find','/kaggle/input','-name','test.csv','-maxdepth','5'], capture_output=True, text=True, timeout=60)
    for line in r.stdout.strip().split('\n'):
        if '0726' in line.lower(): d0726 = str(Path(line).parent); break
assert d0726, '0726 not found'
te726 = pd.read_csv(f'{d0726}/test.csv', usecols=['sample_id','month']).sort_values('sample_id').reset_index(drop=True)
month_te = te726['month'].values
assert len(month_te) == Xte.shape[0]
print('test months:', np.unique(month_te), flush=True)

import lightgbm as lgb
rng = np.random.default_rng(42)
sub = rng.choice(len(Xtr), 200000, replace=False)
dt = lgb.Dataset(Xtr[sub], y_all[sub])
mt = lgb.train(dict(objective='regression', learning_rate=0.1, num_leaves=63, min_child_samples=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbose=-1, seed=42, n_jobs=4),
               dt, num_boost_round=100)
top30 = np.argsort(mt.feature_importance('gain'))[::-1][:30]
top15 = top30[:15]
del dt, mt, sub
np.save('/kaggle/working/xs_top30.npy', top30)
print('top30 selected', top30.tolist(), flush=True)

def xs_block(X, month):
    n = len(X)
    pct = np.zeros((n, 30), np.float32)
    zs = np.zeros((n, 15), np.float32)
    for m in np.unique(month):
        mk = month == m
        Xm = X[mk]
        cnt = mk.sum()
        for i, f in enumerate(top30):
            v = Xm[:, f]
            r = np.empty(cnt, np.float32)
            r[np.argsort(v)] = np.arange(cnt, dtype=np.float32)
            pct[mk, i] = r / max(cnt - 1, 1) - 0.5
            if i < 15:
                zs[mk, i] = (v - v.mean()) / (v.std() + 1e-8)
    sel = X[:, top30]
    o = np.argsort(sel, axis=1)
    cr = np.empty((n, 30), np.float32)
    cr[np.arange(n)[:, None], o] = (np.arange(30, dtype=np.float32) / 29.0 - 0.5)[None, :]
    return np.concatenate([pct, zs, cr], axis=1)

XS_tr = xs_block(Xtr, month_tr)
print('XS_train done', XS_tr.shape, f'{time.time()-t0:.0f}s', flush=True)
np.save('/kaggle/working/XS_train.npy', XS_tr)
del XS_tr
XS_te = xs_block(Xte, month_te)
np.save('/kaggle/working/XS_test.npy', XS_te)
np.save('/kaggle/working/month_test.npy', month_te)
print('XS_test done', XS_te.shape, f'{time.time()-t0:.0f}s', flush=True)
print('DONE', flush=True)
