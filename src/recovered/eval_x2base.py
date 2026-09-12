import numpy as np, gc
import pyarrow.feather as feather
import lightgbm as lgb
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
y = lab.target.values.astype(np.float64); month = lab.month.values
tr = month<=60; va = month>=61
X = np.load('/tmp/work/X2_train.npy'); keys=list(np.load('/tmp/work/X2_train_keys.npy'))
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def cos_eval(p,d): return 'cosine', cos(p.astype(np.float64), d.get_label()), True
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
P0 = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
          feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
          num_threads=2, verbose=-1)
dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=keys)
dva = lgb.Dataset(X[va], label=y[va], feature_name=keys, reference=dtr)
vp=[]
for seed in (7,42,123):
    m = lgb.train(dict(P0, seed=seed), dtr, num_boost_round=3000, valid_sets=[dva],
                  feval=cos_eval, callbacks=[lgb.early_stopping(150, first_metric_only=True)])
    p = m.predict(X[va], num_iteration=m.best_iteration)
    vp.append(p); print(f'seed {seed}: {cos(p,y[va]):.6f} ({m.best_iteration} it)', flush=True)
    del m; gc.collect()
avg = np.mean(vp,0)
print('REBUILT X2 BASELINE seedavg:', round(cos(avg,y[va]),6), 'centered:', round(cos(unit(avg),y[va]),6), flush=True)
np.save('/tmp/work/x2base_seedavg_val.npy', avg)
print('BASE_DONE', flush=True)