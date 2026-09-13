# Pseudo-labeling simulation: can self-training on forward months help transfer?
# Baseline A: train 0-50, eval 61-70. Variant B: train 0-50 + pseudo-labeled 51-60 (weighted), eval 61-70.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy')
tr0 = month<=50; pseudo = (month>=51)&(month<=60); va = month>=61; late = month>=66
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
def train_eval(Xtr, ytr, wtr, seeds=(7,42)):
    preds = []
    for seed in seeds:
        p = dict(params); p['seed']=seed
        ds = lgb.Dataset(Xtr, ytr, weight=wtr)
        dsv = lgb.Dataset(X[va], yv, reference=ds)
        m = lgb.train(p, ds, num_boost_round=3000, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(m.predict(X[va], num_iteration=m.best_iteration))
        del ds, dsv, m; gc.collect()
    return np.mean(preds,0)
# baseline A
pvA = train_eval(X[tr0], y[tr0], None)
u = pvA-pvA.mean()
print(f'A train0-50: full {cos(pvA,yv):.6f} late {cos(u[late[va]], yv[late[va]]):.6f}', flush=True)
res = {'A': dict(full=round(cos(pvA,yv),6), late=round(cos(u[late[va]],yv[late[va]]),6))}
# pseudo-label 51-60 with model trained on 0-50 (seed 7 ensemble of the 2 we already train: reuse pvA-style model on pseudo)
# train a fresh 2-seed model to generate pseudo labels
Xp = X[pseudo]
ppreds = []
for seed in (7,42):
    p = dict(params); p['seed']=seed
    ds = lgb.Dataset(X[tr0], y[tr0])
    m = lgb.train(p, ds, num_boost_round=800)
    ppreds.append(m.predict(Xp))
    del ds, m; gc.collect()
plabel = np.mean(ppreds,0); del ppreds; gc.collect()
print('pseudo label std:', round(float(plabel.std()),8), 'true y std (51-60):', round(float(y[pseudo].std()),8), flush=True)
# variant B: add pseudo rows with weight w0, confident subset only (|plabel| above median)
conf = np.abs(plabel) >= np.median(np.abs(plabel))
for w0 in (0.1, 0.3):
    sel = conf
    Xtr = np.vstack([X[tr0], Xp[sel]])
    ytr = np.concatenate([y[tr0], plabel[sel]])
    wtr = np.concatenate([np.ones(tr0.sum()), np.full(sel.sum(), w0)])
    pvB = train_eval(Xtr, ytr, wtr)
    u = pvB-pvB.mean()
    r = dict(full=round(cos(pvB,yv),6), late=round(cos(u[late[va]],yv[late[va]]),6))
    res[f'B_w{w0}'] = r
    print(f'B pseudo w={w0}: full {r["full"]:.6f} late {r["late"]:.6f}', flush=True)
    del Xtr, ytr, wtr, pvB, u; gc.collect()
# sanity reference: oracle - train 0-60 (uses real labels of 51-60)
pvC = train_eval(X[month<=60], y[month<=60], None)
u = pvC-pvC.mean()
print(f'C oracle 0-60: full {cos(pvC,yv):.6f} late {cos(u[late[va]], yv[late[va]]):.6f}', flush=True)
res['C_oracle'] = dict(full=round(cos(pvC,yv),6), late=round(cos(u[late[va]],yv[late[va]]),6))
json.dump(res, open('/tmp/work/pseudo_results.json','w'), indent=1)
print('PSEUDO_DONE', flush=True)
