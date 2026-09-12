"""Champion model (GBM v4): LightGBM on the 58 per-sample features.

Recipe that produced public leaderboard score 0.110 (validation cosine 0.1274):
  1. bagging_fraction 0.7 (best of a 13-config coordinate sweep)
  2. early stopping on validation months 61-70
  3. average of 3 seeds (7/42/123)
  4. refit on months 0-70 with 1.1x the early-stopped iteration counts
  5. mean-centered predictions (the target is approximately mean-zero, so
     centering the prediction vector raises cosine for free)
  6. zero predictions for the 17 test samples with no underlying data
  7. clip to the 0.1/99.9 percentile band of train predictions

Usage: python3 train_model.py <data_dir> <work_dir>
"""
import sys, time, gc
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lightgbm as lgb

data_dir, work_dir = sys.argv[1], sys.argv[2]

X = np.load(f'{work_dir}/X2_train.npy')
keys = list(np.load(f'{work_dir}/X2_train_keys.npy'))
lab = feather.read_table(f'{data_dir}/train/label.feather').to_pandas()
lab = lab.sort_values('sample_id').reset_index(drop=True)
assert (lab.sample_id.values == np.arange(len(lab))).all()
y = lab.target.values.astype(np.float64)
month = lab.month.values
tr = month <= 60
va = month >= 61

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def cos_eval(preds, dset):
    return 'cosine', cos(preds.astype(np.float64), dset.get_label()), True

def unit(v):
    v = v - v.mean()
    s = v.std()
    return v / (s if s > 0 else 1)

PARAMS = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1)

dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=keys, free_raw_data=False)
dva = lgb.Dataset(X[va], label=y[va], feature_name=keys, reference=dtr, free_raw_data=False)

val_preds, best_iters = [], []
for seed in (7, 42, 123):
    m = lgb.train(dict(PARAMS, seed=seed), dtr, num_boost_round=3000,
                  valid_sets=[dva], feval=cos_eval,
                  callbacks=[lgb.early_stopping(150, first_metric_only=True)])
    val_preds.append(m.predict(X[va], num_iteration=m.best_iteration))
    best_iters.append(m.best_iteration)
    print(f'seed {seed}: val cosine {cos(val_preds[-1], y[va]):.6f} '
          f'({m.best_iteration} iters)', flush=True)
    del m; gc.collect()

avg = np.mean(val_preds, axis=0)
print('seed-average val cosine:', round(cos(avg, y[va]), 6), flush=True)
print('seed-average, mean-centered:', round(cos(unit(avg), y[va]), 6), flush=True)

# refit on all labelled months and predict test
Xte = np.load(f'{work_dir}/X2_test.npy')
kin = {k: i for i, k in enumerate(keys)}
nodata = (Xte[:, kin['tx_n']] == 0) & (Xte[:, kin['mk_nbars']] == 0)
dall = lgb.Dataset(X, label=y, feature_name=keys, free_raw_data=False)
test_pred, train_pred = np.zeros(len(Xte)), np.zeros(len(X))
for seed, it in zip((7, 42, 123), best_iters):
    mf = lgb.train(dict(PARAMS, seed=seed), dall, num_boost_round=int(it * 1.1))
    test_pred += mf.predict(Xte)
    train_pred += mf.predict(X)
    del mf; gc.collect()
test_pred = unit(test_pred / 3)
train_pred = unit(train_pred / 3)

test_pred[nodata] = 0.0
lo, hi = np.quantile(train_pred, [0.001, 0.999])
test_pred = np.clip(test_pred, lo, hi)

sub = pd.read_csv(f'{data_dir}/submission.csv')
sub['prediction'] = test_pred
sub.to_csv(f'{work_dir}/submission.csv', index=False)
print('submission written:', sub.shape, flush=True)
