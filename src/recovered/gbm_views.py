# Two decorrelation attempts in one process:
# V1: GBM on rank-within-month target (X2 features)
# V2: GBM on fingerprint view (fpR logdiffs + fpV log1p + X6 = 63 feats)
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
tr = month<=60; va = month>=61; late = month>=66
yv = y[va]; lv = late[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
def train2(Xtr, ytr, Xv):
    preds=[]
    for seed in (7,42):
        p=dict(params); p['seed']=seed
        ds=lgb.Dataset(Xtr,ytr); dsv=lgb.Dataset(Xv,yv,reference=ds)
        m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])
        preds.append(m.predict(Xv,num_iteration=m.best_iteration)); del ds,dsv,m; gc.collect()
    return np.mean(preds,0)
g = unit(np.load('x2base_seedavg_val.npy')); ml = unit(np.load('mlp_seedavg_val.npy'))
def report(tag, pv):
    u = unit(pv)
    print(f'{tag} solo: full {cos(pv,yv):.6f} late {cos(u[lv],yv[lv]):.6f} corr_gbm {float(np.corrcoef(u,g)[0,1]):.4f} corr_mlp {float(np.corrcoef(u,ml)[0,1]):.4f}', flush=True)
    for wv in (0.15, 0.25):
        bl = unit(0.5*(1-wv)*g + 0.5*(1-wv)*ml + wv*u)
        print(f'  blend+{tag} w={wv}: full {cos(bl,yv):.6f} late {cos(bl[lv],yv[lv]):.6f}', flush=True)
# V1: rank target
X = np.load('X2_train.npy')
yrank = np.empty_like(y)
for mo in np.unique(month[tr]):
    sel = tr & (month==mo)
    r = y[sel].argsort().argsort().astype(np.float64)
    yrank[sel] = r/ (sel.sum()-1) - 0.5
pv1 = train2(X[tr], yrank[tr], X[va])
np.save('gbm_rank_val.npy', pv1)
report('V1_ranktarget', pv1)
del X, pv1, yrank; gc.collect()
# V2: fingerprint view
NG=30
fpR = np.load('fpR_train.npy'); fpR = np.maximum(fpR,1e-9); np.log(fpR,out=fpR); fpRd = np.diff(fpR,axis=1); del fpR
fpV = np.load('fpV_train.npy'); np.log1p(fpV,out=fpV)
X6 = np.load('X6_train.npy')
Xf = np.concatenate([fpRd, fpV, X6], axis=1).astype(np.float32)
del fpRd, fpV, X6; gc.collect()
np.nan_to_num(Xf, copy=False)
pv2 = train2(Xf[tr], y[tr], Xf[va])
np.save('gbm_fpview_val.npy', pv2)
report('V2_fpview', pv2)
print('VIEWS_DONE', flush=True)
