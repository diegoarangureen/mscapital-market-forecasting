# Iterative adversarial pruning (low RAM via mmap): late-train(66-70) vs test on 303f.
import numpy as np, gc, json, time
import lightgbm as lgb
t0=time.time()
WORK='/tmp/work'
month = np.load(f'{WORK}/full_month.npy')
names = np.load(f'{WORK}/full_names.npy', allow_pickle=True).tolist()
n21 = np.load(f'{WORK}/X21_train_names.npy', allow_pickle=True).tolist()
n22 = np.load(f'{WORK}/X22_train_names.npy', allow_pickle=True).tolist()
allnames = names + n21 + n22
late_idx = np.where(month >= 66)[0]
def rows(split, idx=None):
    parts = []
    for f in ['full', 'X21', 'X22']:
        m = np.load(f'{WORK}/{f}_{split}.npy', mmap_mode='r')
        a = np.asarray(m[idx] if idx is not None else m[:], dtype=np.float32)
        parts.append(np.nan_to_num(a))
    return np.concatenate(parts, axis=1)
Xl = rows('train', late_idx)
rng = np.random.default_rng(0)
te_sub = np.sort(rng.choice(647896, 300000, replace=False))
Xte = rows('test', te_sub)
print('late', Xl.shape, 'test-sub', Xte.shape, flush=True)
Xc = np.vstack([Xl, Xte]); del Xl, Xte; gc.collect()
yc = np.concatenate([np.zeros(len(late_idx)), np.ones(len(te_sub))])
def auc_of(p, y):
    r = p.argsort().argsort().astype(np.float64)
    pos = y==1
    return float((r[pos].sum() - pos.sum()*(pos.sum()-1)/2) / (pos.sum()*(~pos).sum()))
def fit_auc(cols):
    ds = lgb.Dataset(Xc[:, cols], yc)
    pc = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=2)
    m = lgb.train(pc, ds, num_boost_round=400)
    p = m.predict(Xc[:, cols])
    a = auc_of(p, yc)
    imp = m.feature_importance('gain')
    del ds, m; gc.collect()
    return a, imp
all_cols = np.arange(Xc.shape[1])
a0, imp0 = fit_auc(all_cols)
print('AUC full 303f:', round(a0,4), flush=True)
order = np.argsort(-imp0)
res = {'auc_full': a0, 'ranked_idx': order.tolist(), 'ranked_names': [allnames[i] for i in order]}
for k in (5, 10, 15, 20, 30):
    keep = np.setdiff1d(all_cols, order[:k])
    a, _ = fit_auc(keep)
    res[f'auc_drop{k}'] = a
    print(f'drop {k}: AUC {a:.4f} dropped={[allnames[i] for i in order[:k]]}', flush=True)
json.dump(res, open(f'{WORK}/adv_iter.json','w'), indent=1)
print('ADVITER_DONE', round(time.time()-t0), 's', flush=True)
