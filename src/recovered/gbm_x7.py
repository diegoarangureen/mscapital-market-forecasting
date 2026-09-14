# GBM on X2+X7 (94 feats): does bucketed L2 add signal? 2 seeds, early stop, full+late+blend.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X2 = np.load('/tmp/work/X2_train.npy')
X7 = np.load('/tmp/work/X7_train.npy')
X = np.concatenate([X2, X7], axis=1); del X2, X7; gc.collect()
np.nan_to_num(X, copy=False)
tr = month<=60; va = month>=61; late = month>=66
yv = y[va]; lv = late[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
preds=[]
for seed in (7,42):
    p=dict(params); p['seed']=seed
    ds=lgb.Dataset(X[tr],y[tr]); dsv=lgb.Dataset(X[va],yv,reference=ds)
    m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])
    preds.append(m.predict(X[va],num_iteration=m.best_iteration))
    print('seed',seed,'best_iter',m.best_iteration,flush=True)
    del ds,dsv,m; gc.collect()
pv=np.mean(preds,0); np.save('/tmp/work/gbm_x7_val.npy', pv)
u=unit(pv)
print(f'GBM X2+X7 solo: full {cos(pv,yv):.6f} late {cos(u[lv],yv[lv]):.6f}', flush=True)
print('(2-seed base reference: full 0.123587 late 0.126569)', flush=True)
g = unit(np.load('/tmp/work/x2base_seedavg_val.npy')); ml = unit(np.load('/tmp/work/mlp_seedavg_val.npy'))
bl = unit(0.5*g+0.5*ml)
print(f'champion blend: full {cos(bl,yv):.6f} late {cos(bl[lv],yv[lv]):.6f}', flush=True)
bl2 = unit(0.45*u+0.55*ml)  # swap in x7 gbm, slight mlp tilt per late-val
print(f'blend gbmX7+mlp 45/55: full {cos(bl2,yv):.6f} late {cos(bl2[lv],yv[lv]):.6f}', flush=True)
bl3 = unit(0.5*0.5*(g+u)/1.0+0.5*ml)
print(f'blend (gbm+gbmX7)/2+mlp: full {cos(bl3,yv):.6f} late {cos(bl3[lv],yv[lv]):.6f}', flush=True)
# feature importance of the X7 block (quick, from last trained model is gone; retrain small)
print('X7_DONE', flush=True)
