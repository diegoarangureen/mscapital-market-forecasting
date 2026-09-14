# GBM on X2+X7+X8+X9 (150 feats) vs X2+X7 (94). 2 seeds, early stop, full+late+blend.
import numpy as np, gc
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
names_ = ('X2','X7','X8','X9')
p0 = np.load(f'/tmp/work/X2_train.npy')
n, d0 = p0.shape
X = np.empty((n, 150), np.float32)
X[:, :d0] = p0; del p0; gc.collect()
col = d0
for nm in ('X7','X8','X9'):
    P = np.load(f'/tmp/work/{nm}_train.npy')
    X[:, col:col+P.shape[1]] = P; col += P.shape[1]; del P; gc.collect()
assert col == 150
for st in range(0, n, 100000):
    blk = X[st:st+100000]
    m_ = ~np.isfinite(blk); blk[m_] = 0.0; del m_

tr = month<=60; va = month>=61; late = month>=66
yv = y[va]; lv = late[va]
# stage row subsets to disk memmaps, then drop the full matrix
import os
tri = np.where(tr)[0]; vai = np.where(va)[0]
Xtr_mm = np.memmap('/tmp/work/x789_tr.f32', dtype=np.float32, mode='w+', shape=(len(tri),150))
Xv_mm = np.memmap('/tmp/work/x789_va.f32', dtype=np.float32, mode='w+', shape=(len(vai),150))
for st in range(0, len(tri), 100000):
    Xtr_mm[st:st+100000] = X[tri[st:st+100000]]
for st in range(0, len(vai), 100000):
    Xv_mm[st:st+100000] = X[vai[st:st+100000]]
Xtr_mm.flush(); Xv_mm.flush()
del X, tri, vai; gc.collect()
print('staged to disk', flush=True)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
preds=[]; iters=[]
for seed in (7,42):
    p=dict(params); p['seed']=seed
    ds=lgb.Dataset(Xtr_mm,y[tr]); dsv=lgb.Dataset(Xv_mm,yv,reference=ds)
    m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])
    preds.append(m.predict(Xv_mm,num_iteration=m.best_iteration)); iters.append(m.best_iteration)
    print('seed',seed,'best_iter',m.best_iteration,flush=True)
    del ds,dsv,m; gc.collect()
pv=np.mean(preds,0); np.save('/tmp/work/gbm_x789_val.npy', pv)
u=unit(pv)
print(f'GBM X2+X7+X8+X9 solo: full {cos(pv,yv):.6f} late {cos(u[lv],yv[lv]):.6f}', flush=True)
print('refs: X2+X7 0.124663/0.127993 | X2 0.123587/0.126569', flush=True)
g = unit(np.load('/tmp/work/gbm_x7_val.npy')); ml = unit(np.load('/tmp/work/mlp_seedavg_val.npy'))
g2 = unit(np.load('/tmp/work/x2base_seedavg_val.npy'))
bl = unit(0.25*g2+0.25*g+0.5*ml)
print(f'current best blend (25/25/50): full {cos(bl,yv):.6f} late {cos(bl[lv],yv[lv]):.6f}', flush=True)
bl2 = unit(0.5*u+0.5*ml)
print(f'blend gbmX789+mlp 50/50: full {cos(bl2,yv):.6f} late {cos(bl2[lv],yv[lv]):.6f}', flush=True)
bl3 = unit(0.2*g2+0.2*g+0.1*u+0.5*ml)
print(f'blend 20/20/10/50: full {cos(bl3,yv):.6f} late {cos(bl3[lv],yv[lv]):.6f}', flush=True)
import json; json.dump(iters, open('/tmp/work/x789_best_iters.json','w'))
print('X789_DONE', flush=True)
