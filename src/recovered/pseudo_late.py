# Shift-robustness of pseudo-labeling: train 0-40, pseudo 41-50, eval 51-60.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy')
tr0 = month<=55; pseudo = (month>=56)&(month<=65); va = month>=66
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
def train_eval(Xtr, ytr, wtr):
    preds = []
    for seed in (7,42):
        p = dict(params); p['seed']=seed
        ds = lgb.Dataset(Xtr, ytr, weight=wtr)
        dsv = lgb.Dataset(X[va], yv, reference=ds)
        m = lgb.train(p, ds, num_boost_round=3000, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(m.predict(X[va], num_iteration=m.best_iteration))
        del ds, dsv, m; gc.collect()
    return np.mean(preds,0)
res = {}
pvA = train_eval(X[tr0], y[tr0], None)
res['A_shift'] = round(cos(pvA,yv),6)
print(f'A late train0-55: val66-70 {res["A_shift"]:.6f}', flush=True)
Xp = X[pseudo]
ppreds = []
for seed in (7,42):
    p = dict(params); p['seed']=seed
    ds = lgb.Dataset(X[tr0], y[tr0])
    m = lgb.train(p, ds, num_boost_round=800)
    ppreds.append(m.predict(Xp))
    del ds, m; gc.collect()
plabel = np.mean(ppreds,0); del ppreds; gc.collect()
conf = np.abs(plabel) >= np.median(np.abs(plabel))
Xtr = np.vstack([X[tr0], Xp[conf]])
ytr = np.concatenate([y[tr0], plabel[conf]])
wtr = np.concatenate([np.ones(tr0.sum()), np.full(conf.sum(), 0.1)])
pvB = train_eval(Xtr, ytr, wtr)
res['B_shift_w0.1'] = round(cos(pvB,yv),6)
print(f'B late pseudo w=0.1: val66-70 {res["B_shift_w0.1"]:.6f}', flush=True)
json.dump(res, open('/tmp/work/pseudo_late_results.json','w'), indent=1)
print('PSEUDO_LATE_DONE', flush=True)
