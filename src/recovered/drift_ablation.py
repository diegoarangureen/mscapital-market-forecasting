# Drop most-drifting features (PSI train-early vs test) and measure late-val impact.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy'); Xt = np.load('/tmp/work/X2_test.npy')
keys = np.load('/tmp/work/X2_train_keys.npy', allow_pickle=True)
tr_all = month<=60; va = month>=61; late = month>=66
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
# PSI per feature: train(0-60) vs test, 10 quantile bins from train
n_tr = tr_all.sum()
psi = np.zeros(X.shape[1])
for j in range(X.shape[1]):
    a = X[tr_all, j]; b = Xt[:, j]
    qs = np.quantile(a, np.linspace(0,1,11)); qs[0]-=1e-9; qs[-1]+=1e-9
    ca = np.histogram(a, qs)[0]/len(a); cb = np.histogram(b, qs)[0]/len(b)
    ca = np.clip(ca,1e-4,None); cb = np.clip(cb,1e-4,None)
    psi[j] = ((ca-cb)*np.log(ca/cb)).sum()
order = np.argsort(-psi)
print('top drift feats:', [(keys[i], round(float(psi[i]),3)) for i in order[:10]], flush=True)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
res = {}
for K in (0, 5, 10, 20):
    keep = np.setdiff1d(np.arange(X.shape[1]), order[:K])
    preds = []
    for seed in (7,42):
        p = dict(params); p['seed']=seed
        ds = lgb.Dataset(X[np.ix_(tr_all, keep)], y[tr_all])
        dsv = lgb.Dataset(X[np.ix_(va, keep)], yv, reference=ds)
        m = lgb.train(p, ds, num_boost_round=3000, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(m.predict(X[np.ix_(va, keep)], num_iteration=m.best_iteration))
        del ds, dsv, m; gc.collect()
    pv = np.mean(preds,0); u = pv-pv.mean()
    res[K] = dict(full=round(cos(pv,yv),6), late=round(cos(u[late[va]], yv[late[va]]),6),
                  dropped=[str(keys[i]) for i in order[:K]])
    print(f'drop{K}: full {res[K]["full"]:.6f} late {res[K]["late"]:.6f}', flush=True)
    del preds, pv, u; gc.collect()
json.dump(res, open('/tmp/work/drift_results.json','w'), indent=1)
print('DRIFT_DONE', flush=True)
