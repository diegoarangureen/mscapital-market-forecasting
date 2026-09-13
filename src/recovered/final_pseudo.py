# Final pseudo-label pipeline for test.
# Pseudo labels: existing 3-seed refit GBM test preds (refit_gbm_test.npy).
# Retrain GBM seeds 7/42/123 on months 0-70 + confident half of test (w=0.1).
# Blend 50/50 with refit MLP test preds, same post-processing as submission_blend.csv.
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy'); Xt = np.load('/tmp/work/X2_test.npy')
iters = json.load(open('/tmp/work/x2base_iters.json'))
print('best iters record:', iters, flush=True)
nround = int(np.mean(iters)*1.1) if isinstance(iters, list) else int(iters['mean_best']*1.1)
print('using nround', nround, flush=True)
pl = np.load('/tmp/work/refit_gbm_test.npy')  # 3-seed avg test preds (units of y)
conf = np.abs(pl) >= np.median(np.abs(pl))
print('confident test rows:', int(conf.sum()), 'of', len(pl), flush=True)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
Xtr = np.vstack([X, Xt[conf]])
ytr = np.concatenate([y, pl[conf]])
wtr = np.concatenate([np.ones(len(y)), np.full(int(conf.sum()), 0.1)])
del X; gc.collect()
preds = []
for seed in (7,42,123):
    p = dict(params); p['seed']=seed
    ds = lgb.Dataset(Xtr, ytr, weight=wtr)
    m = lgb.train(p, ds, num_boost_round=nround)
    preds.append(m.predict(Xt))
    print('seed', seed, 'done', flush=True)
    del ds, m; gc.collect()
pt = np.mean(preds,0)
np.save('/tmp/work/pseudo_gbm_test.npy', pt)
print('PSEUDO_GBM_DONE', flush=True)
