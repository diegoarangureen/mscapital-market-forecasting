# Adversarial importance weighting sim: train 0-50, target distribution = 61-70, eval 61-70.
# Classifier distinguishes train vs forward months; density-ratio weights upweight test-like train rows.
# Baseline for comparison (same split, 2 seeds): A full 0.121617 late 0.125164.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy')
tr0 = month<=50; va = month>=61; late = month>=66
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
# adversarial classifier: train(0-50)=0 vs forward(61-70)=1
Xc = np.vstack([X[tr0], X[va]])
yc = np.concatenate([np.zeros(tr0.sum()), np.ones(va.sum())])
ds = lgb.Dataset(Xc, yc)
pc = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
          feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=2)
mc = lgb.train(pc, ds, num_boost_round=400)
ptr = mc.predict(X[tr0])  # P(forward-like)
pva = mc.predict(X[va])
pc_all = mc.predict(Xc)
rng = np.random.default_rng(0)
sub = rng.choice(len(yc), 200000, replace=False)
r = pc_all[sub].argsort().argsort().astype(np.float64)
pos = yc[sub]==1
auc = (r[pos].sum() - pos.sum()*(pos.sum()-1)/2) / (pos.sum()*(~pos).sum())
print('adversarial AUC (train vs forward):', round(float(auc),4), flush=True)
del Xc, yc, ds, mc; gc.collect()
w = ptr/(1-ptr+1e-6); w = np.clip(w, 0, np.quantile(w, 0.99))
w = w/w.mean()
print('weight stats: mean1, p50', round(float(np.median(w)),3), 'p99', round(float(np.quantile(w,0.99)),3), flush=True)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
for wtag, wuse in (('adv', w), ('none', None)):
    preds = []
    for seed in (7,42):
        p = dict(params); p['seed']=seed
        d = lgb.Dataset(X[tr0], y[tr0], weight=wuse)
        dsv = lgb.Dataset(X[va], yv, reference=d)
        m = lgb.train(p, d, num_boost_round=3000, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(m.predict(X[va], num_iteration=m.best_iteration))
        del d, dsv, m; gc.collect()
    pv = np.mean(preds,0); u = pv-pv.mean()
    print(f'{wtag}: full {cos(pv,yv):.6f} late {cos(u[late[va]], yv[late[va]]):.6f}', flush=True)
    json.dump({'auc': float(auc)}, open('/tmp/work/adv_auc.json','w'))
print('ADV_DONE', flush=True)
