# Shift-robustness: train 0-50, val 51-60. X2-only ref was 0.122894 (shift_gbm_val.npy).
import numpy as np, gc
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X2 = np.load('/tmp/work/X2_train.npy'); X7 = np.load('/tmp/work/X7_train.npy')
X = np.concatenate([X2, X7], axis=1); del X2, X7; gc.collect()
np.nan_to_num(X, copy=False)
tr = month<=50; va = (month>=51)&(month<=60)
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
preds=[]
for seed in (7,42):
    p=dict(params); p['seed']=seed
    ds=lgb.Dataset(X[tr],y[tr]); dsv=lgb.Dataset(X[va],yv,reference=ds)
    m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])
    preds.append(m.predict(X[va],num_iteration=m.best_iteration))
    del ds,dsv,m; gc.collect()
pv=np.mean(preds,0)
print(f'SHIFT X2+X7: {cos(pv,yv):.6f}  (X2-only shift ref: 0.122894)', flush=True)
np.save('/tmp/work/shift_x7_val.npy', pv)
print('X7SHIFT_DONE', flush=True)
