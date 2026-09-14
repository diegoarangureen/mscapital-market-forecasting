# Refit GBM X2+X7 on months 0-70, 3 seeds, 1.1x best iters -> test + train preds.
import numpy as np, gc, json, sys
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)
del lab; gc.collect()
X2 = np.load('/tmp/work/X2_train.npy'); X7 = np.load('/tmp/work/X7_train.npy')
X = np.concatenate([X2, X7], axis=1); del X2, X7; gc.collect()
np.nan_to_num(X, copy=False)
X2t = np.load('/tmp/work/X2_test.npy'); X7t = np.load('/tmp/work/X7_test.npy')
Xt = np.concatenate([X2t, X7t], axis=1); del X2t, X7t; gc.collect()
np.nan_to_num(Xt, copy=False)
iters = [int(x*1.1) for x in json.load(open('/tmp/work/x7_best_iters.json'))]
print('refit iters:', iters, flush=True)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
which = sys.argv[1] if len(sys.argv)>1 else 'test'
if which=='test':
    preds=[]
    for seed, nr in zip((7,42,123), iters):
        p=dict(params); p['seed']=seed
        ds=lgb.Dataset(X, y)
        m=lgb.train(p, ds, num_boost_round=nr)
        preds.append(m.predict(Xt))
        print('test seed',seed,'done',flush=True)
        del ds, m; gc.collect()
    np.save('/tmp/work/refit_x7_test.npy', np.mean(preds,0))
    print('X7_REFIT_TEST_DONE', flush=True)
else:
    preds=[]
    for seed, nr in zip((7,42,123), iters):
        p=dict(params); p['seed']=seed
        ds=lgb.Dataset(X, y)
        m=lgb.train(p, ds, num_boost_round=nr)
        preds.append(m.predict(X))
        print('train seed',seed,'done',flush=True)
        del ds, m; gc.collect()
    np.save('/tmp/work/refit_x7_train.npy', np.mean(preds,0))
    print('X7_REFIT_TRAIN_DONE', flush=True)
