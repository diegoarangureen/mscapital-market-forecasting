import numpy as np, pyarrow.feather as feather, pandas as pd, time, gc, json, sys
import lightgbm as lgb

XPREFIX = sys.argv[1] if len(sys.argv)>1 else 'X2'   # X2 or X3
X = np.load(f'/tmp/work/{XPREFIX}_train.npy')
keys = list(np.load(f'/tmp/work/{XPREFIX}_train_keys.npy'))
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
tr = month <= 60; va = month >= 61
def cos(a,b): return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def cos_eval(preds, dset):
    return 'cosine', cos(preds.astype(np.float64), dset.get_label()), True
def unit(v):
    v = v - v.mean(); s = v.std(); return v/(s if s>0 else 1)

BASE = dict(objective='regression', learning_rate=0.05, num_leaves=127,
            min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=7)
OVERRIDES = {
 'base': {}, 'lr0.03': {'learning_rate':0.03}, 'lr0.08': {'learning_rate':0.08},
 'leaves63': {'num_leaves':63}, 'leaves255': {'num_leaves':255},
 'mindata200': {'min_data_in_leaf':200}, 'mindata1000': {'min_data_in_leaf':1000},
 'ff0.7': {'feature_fraction':0.7}, 'ff0.9': {'feature_fraction':0.9},
 'l2_0': {'lambda_l2':0.0}, 'l2_10': {'lambda_l2':10.0},
 'bag0.7': {'bagging_fraction':0.7}, 'nobag': {'bagging_fraction':1.0,'bagging_freq':0},
}
meta = json.load(open('/tmp/work/ens_meta.json'))
blend = json.load(open('/tmp/work/ens_blend.json'))
ridgej = json.load(open('/tmp/work/ens_ridge.json'))
best_name = meta['best_name']; iters = meta['iters']
w_gbm = blend['blend'][0][1]   # best val blend weight
alpha = ridgej['ridge_alpha']
print('final config:', best_name, 'iters', iters, 'w_gbm', w_gbm, 'ridge_alpha', alpha, flush=True)

Xte = np.load(f'/tmp/work/{XPREFIX}_test.npy')
kin = {k:i for i,k in enumerate(keys)}
nodata = (Xte[:, kin['tx_n']]==0) & (Xte[:, kin['mk_nbars']]==0)
print('no-data test samples:', int(nodata.sum()), flush=True)

dall = lgb.Dataset(X, label=y, feature_name=keys, free_raw_data=False)
gbm_test = np.zeros(len(Xte)); gbm_train = np.zeros(len(X))
for seed, it in zip((7,42,123), iters):
    p = dict(BASE); p.update(OVERRIDES[best_name]); p['seed']=seed
    n_it = int(it*1.1)
    t0=time.time()
    mf = lgb.train(p, dall, num_boost_round=n_it, callbacks=[lgb.log_evaluation(500)])
    gbm_test += mf.predict(Xte); gbm_train += mf.predict(X)
    print(f'seed {seed} refit {n_it} iters {round(time.time()-t0)}s', flush=True)
    del mf; gc.collect()
gbm_test/=3; gbm_train/=3
print('gbm train cos (all months):', round(cos(gbm_train, y),6), flush=True)
del dall; gc.collect()

Xf = np.concatenate([X, Xte]).astype(np.float64)
del X, Xte; gc.collect()
mu = Xf[:len(y)].mean(0); sd = Xf[:len(y)].std(0); sd[sd==0]=1
Z = np.nan_to_num((Xf-mu)/sd); del Xf; gc.collect()
Ztr = Z[:len(y)]; Zte = Z[len(y):]; del Z; gc.collect()
w = np.linalg.solve(Ztr.T @ Ztr + alpha*np.eye(Ztr.shape[1]), Ztr.T @ y)
ridge_test = Zte @ w; ridge_train = Ztr @ w
del Ztr, Zte; gc.collect()
print('ridge train cos (all):', round(cos(ridge_train, y),6), flush=True)

pred = w_gbm*unit(gbm_test) + (1-w_gbm)*unit(ridge_test)
ptr  = w_gbm*unit(gbm_train) + (1-w_gbm)*unit(ridge_train)
print('blend train cos (all):', round(cos(ptr, y),6), flush=True)
pred[nodata] = 0.0
lo, hi = np.quantile(ptr, [0.001, 0.999])
pred = np.clip(pred, lo, hi)
sub = pd.read_csv('/tmp/mscapital/submission.csv')
assert len(sub)==len(pred) and (sub.sample_id.values==np.arange(len(sub))).all()
sub['prediction'] = pred
sub.to_csv('/tmp/work/submission_final.csv', index=False)
print('submission_final.csv written', sub.shape, 'pred std', float(np.std(pred)), flush=True)