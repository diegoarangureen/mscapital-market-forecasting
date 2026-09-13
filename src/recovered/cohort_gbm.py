# Pivot: GBM trained on recent-month cohorts vs full data.
# Motivation: val->LB transfer tracks late-month regime; older months may hurt.
import numpy as np, gc, time, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy')
tr_all = month<=60; va = month>=61; late = month>=66
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
res = {}
for lo in (0, 31, 41, 46, 51):
    tr = tr_all & (month>=lo)
    preds = []
    for seed in (7,42):
        p = dict(params); p['seed']=seed
        ds = lgb.Dataset(X[tr], y[tr])
        dsv = lgb.Dataset(X[va], yv, reference=ds)
        m = lgb.train(p, ds, num_boost_round=3000, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(m.predict(X[va], num_iteration=m.best_iteration))
        del ds, dsv, m; gc.collect()
    pv = np.mean(preds,0); u = pv-pv.mean()
    res[lo] = dict(full=round(cos(pv,yv),6), late=round(cos(u[late[va]], yv[late[va]]),6))
    print(f'cohort {lo}-60: full {res[lo]["full"]:.6f} late {res[lo]["late"]:.6f}', flush=True)
    del preds, pv, u; gc.collect()
np.save('/tmp/work/cohort_preds_val.npy', np.zeros(1))  # placeholder
json.dump(res, open('/tmp/work/cohort_results.json','w'))
print('COHORT_DONE', flush=True)
