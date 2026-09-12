# robustness: train 0-50, val 51-60, GBM + MLP + 50/50 blend
import numpy as np, gc
import pyarrow.feather as feather
import lightgbm as lgb
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
y = lab.target.values.astype(np.float64); month = lab.month.values
tr = month<=50; va = (month>=51)&(month<=60)
X = np.load('/tmp/work/X2_train.npy'); keys=list(np.load('/tmp/work/X2_train_keys.npy'))
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def cos_eval(p,d): return 'cosine', cos(p.astype(np.float64), d.get_label()), True
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
P0 = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
          feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
          num_threads=2, verbose=-1)
yv = y[va]
dva = lgb.Dataset(X[va], label=y[va], feature_name=keys)
vp=[]
for seed in (7,42,123):
    dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=keys)
    m = lgb.train(dict(P0, seed=seed), dtr, num_boost_round=3000, valid_sets=[dva],
                  feval=cos_eval, callbacks=[lgb.early_stopping(150, first_metric_only=True)])
    vp.append(m.predict(X[va], num_iteration=m.best_iteration))
    print(f'gbm seed {seed}: {cos(vp[-1],yv):.6f}', flush=True)
    del m, dtr; gc.collect()
g = np.mean(vp,0)
print('SHIFT GBM seedavg:', round(cos(g,yv),6), 'centered:', round(cos(unit(g),yv),6), flush=True)
np.save('/tmp/work/shift_gbm_val.npy', g)
del X, dva; gc.collect()
print('GBM_PART_DONE', flush=True)