# GBM on X2+X7+X8+X9+X10 (210 feats). 2 seeds, early stop, full+late+blend evals.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
n = len(y); D = 210
tr = month<=60; va = month>=61; late = month>=66
yv = y[va]; lv = late[va]
ntr = int(tr.sum()); nva = int(va.sum())
Xtr = np.memmap('/tmp/work/x10_tr.f32', dtype=np.float32, mode='w+', shape=(ntr, D))
Xv  = np.memmap('/tmp/work/x10_va.f32', dtype=np.float32, mode='w+', shape=(nva, D))
col = 0
for nm in ('X2','X7','X8','X9','X10'):
    P = np.load(f'/tmp/work/{nm}_train.npy')
    d = P.shape[1]
    ctr = 0; cva = 0
    for st in range(0, n, 100000):
        blk = P[st:st+100000].astype(np.float32, copy=False)
        m_ = ~np.isfinite(blk); blk[m_] = 0.0; del m_
        sb = tr[st:st+100000]; vb = va[st:st+100000]
        k1 = int(sb.sum()); k2 = int(vb.sum())
        if k1: Xtr[ctr:ctr+k1, col:col+d] = blk[sb]
        if k2: Xv[cva:cva+k2, col:col+d] = blk[vb]
        ctr += k1; cva += k2
        del blk
    col += d; del P; gc.collect()
    print(nm, 'staged, col', col, flush=True)
assert col == D and ctr == ntr and cva == nva
Xtr.flush(); Xv.flush()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
preds=[]; iters=[]
for seed in (7,42):
    p=dict(params); p['seed']=seed
    ds=lgb.Dataset(Xtr,y[tr]); dsv=lgb.Dataset(Xv,yv,reference=ds)
    m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])
    preds.append(m.predict(Xv,num_iteration=m.best_iteration)); iters.append(m.best_iteration)
    print('seed',seed,'best_iter',m.best_iteration,flush=True)
    del ds,dsv,m; gc.collect()
pv=np.mean(preds,0); np.save('/tmp/work/gbm_x10_val.npy', pv)
u=unit(pv)
print(f'GBM X2+X789+X10 solo: full {cos(pv,yv):.6f} late {cos(u[lv],yv[lv]):.6f}', flush=True)
print('refs: X789 0.125xxx | X2+X7 0.124663/0.127993 | X2 0.123587/0.126569', flush=True)
ml = unit(np.load('/tmp/work/mlp_seedavg_val.npy'))
bl2 = unit(0.5*u+0.5*ml)
print(f'blend gbmX10+mlp 50/50: full {cos(bl2,yv):.6f} late {cos(bl2[lv],yv[lv]):.6f}', flush=True)
gx = unit(np.load('/tmp/work/gbm_x789_val.npy'))
bl3 = unit(0.25*u+0.25*gx+0.5*ml)
print(f'blend X10/X789/mlp 25/25/50: full {cos(bl3,yv):.6f} late {cos(bl3[lv],yv[lv]):.6f}', flush=True)
json.dump(iters, open('/tmp/work/x10_best_iters.json','w'))
print('X10_DONE', flush=True)
