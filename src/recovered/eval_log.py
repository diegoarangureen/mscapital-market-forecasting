# E3: log1p-robustify drifting count/activity features, champion GBM params, 3 seeds
import numpy as np, gc, json
import pyarrow.feather as feather
import lightgbm as lgb
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
tr = month<=60; va = month>=61; late = month>=66
X = np.load('/tmp/work/X2_train.npy'); keys=list(np.load('/tmp/work/X2_train_keys.npy'))
COUNT_LIKE = ['tx_n','tx_vol','tx_avg_trade','tx_max_trade','mk_cnt','mk_cnt_total','mk_vol',
              'mk_vol_total','mk_vol_last60','mk_vol_60_300','ord_nnew','ord_ncan','mk_px_vol']
idx = [keys.index(k) for k in COUNT_LIKE if k in keys]
idx = [i for i in idx if X[:, i].min() >= -1e-6]
print('log1p applied to:', [keys[i] for i in idx], flush=True)
Xl = X.copy()
for i in idx:
    Xl[:, i] = np.log1p(np.maximum(Xl[:, i], 0))
del X; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def cos_eval(p,d): return 'cosine', cos(p.astype(np.float64), d.get_label()), True
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
P0 = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
          feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
          num_threads=2, verbose=-1)
dtr = lgb.Dataset(Xl[tr], label=y[tr], feature_name=keys)
dva = lgb.Dataset(Xl[va], label=y[va], feature_name=keys, reference=dtr)
vp=[]
for seed in (7,42,123):
    m = lgb.train(dict(P0, seed=seed), dtr, num_boost_round=3000, valid_sets=[dva],
                  feval=cos_eval, callbacks=[lgb.early_stopping(150, first_metric_only=True)])
    vp.append(m.predict(Xl[va], num_iteration=m.best_iteration))
    print(f'seed {seed}: {cos(vp[-1],y[va]):.6f} ({m.best_iteration} it)', flush=True)
    del m; gc.collect()
avg = np.mean(vp,0)
u = unit(avg)
lv = late[va]
print('LOGFEAT seedavg full:', round(cos(avg,y[va]),6), 'centered:', round(cos(u,y[va]),6), flush=True)
print('LOGFEAT seedavg LATE (66-70):', round(cos(u[lv], y[va][lv]),6), flush=True)
print('reference: baseline full 0.126330 centered, late 0.128253', flush=True)
np.save('/tmp/work/logfeat_seedavg_val.npy', avg)
print('E3_DONE', flush=True)
