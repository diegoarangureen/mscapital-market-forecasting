import numpy as np, pyarrow.feather as feather, pandas as pd, time, gc
import lightgbm as lgb

X = np.load('/tmp/work/X2_train.npy')
keys = list(np.load('/tmp/work/X2_train_keys.npy'))
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas()
lab = lab.sort_values('sample_id').reset_index(drop=True)
assert (lab.sample_id.values == np.arange(len(lab))).all()
y = lab.target.values.astype(np.float64); month = lab.month.values

def cos(a, b): return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-30))

tr = month <= 60; va = month >= 61

# --- reproduce ridge baseline on the 22 baseline features ---
BASE = sorted(['mk_avgpx_dev','mk_cnt','mk_imb1','mk_imb2','mk_ret10m','mk_spread','mk_vol',
 'ord_buy_px','ord_cancel_ratio','ord_net_imb','ord_new_imb','ord_ncan','ord_nnew','ord_sell_px',
 'tx_firstp','tx_imb','tx_lastp','tx_n','tx_ret','tx_vol','tx_vwap','tx_vwap_dev'])
bidx = [keys.index(k) for k in BASE]
Xb = X[:, bidx].astype(np.float64)
mu = Xb[tr].mean(0); sd = Xb[tr].std(0); sd[sd==0]=1
Z = np.nan_to_num((Xb-mu)/sd)
best = (-1, None)
for alpha in [1e2, 1e3, 1e4, 1e5, 1e6]:
    A = Z[tr].T @ Z[tr] + alpha*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T @ y[tr])
    c = cos(Z[va] @ w, y[va])
    print(f'ridge alpha={alpha:g} val_cos={c:.6f}', flush=True)
    if c > best[0]: best = (c, alpha)
print('ridge best', best, flush=True)
del Xb, Z, A, w; gc.collect()

# --- LightGBM on all features ---
def cos_eval(preds, dset):
    yy = dset.get_label()
    return 'cosine', cos(preds.astype(np.float64), yy), True

dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=keys, free_raw_data=False)
dva = lgb.Dataset(X[va], label=y[va], feature_name=keys, reference=dtr, free_raw_data=False)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=7)
t0=time.time()
m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
              feval=cos_eval, callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(100)])
print('train time', round(time.time()-t0), 'best_iter', m.best_iteration, flush=True)
pv = m.predict(X[va], num_iteration=m.best_iteration)
val_cos = cos(pv, y[va])
print('LGBM val cosine:', round(val_cos,6), flush=True)
imp = sorted(zip(keys, m.feature_importance('gain')), key=lambda t:-t[1])[:20]
print('top features:', [(k, int(v)) for k,v in imp], flush=True)
np.save('/tmp/work/lgb_val_pred.npy', pv)

VAL_THRESHOLD = 0.0627
if val_cos <= VAL_THRESHOLD:
    print(f'GBM does NOT beat baseline ({val_cos:.6f} <= {VAL_THRESHOLD}) - will not submit', flush=True)
    raise SystemExit(0)

# --- refit on all months with best iteration count, predict test ---
Xte = np.load('/tmp/work/X2_test.npy')
dall = lgb.Dataset(X, label=y, feature_name=keys, free_raw_data=False)
final_iter = int(m.best_iteration * 1.1)
mf = lgb.train(params, dall, num_boost_round=final_iter, callbacks=[lgb.log_evaluation(200)])
pred = mf.predict(Xte)
ptr = mf.predict(X)
print('train cosine (all):', round(cos(ptr, y),6), flush=True)
# zero no-data test samples: no trades and no market bars
kin = {k:i for i,k in enumerate(keys)}
nodata = (Xte[:, kin['tx_n']]==0) & (Xte[:, kin['mk_nbars']]==0)
print('no-data test samples:', int(nodata.sum()), flush=True)
pred[nodata] = 0.0
lo, hi = np.quantile(ptr, [0.001, 0.999])
pred = np.clip(pred, lo, hi)
sub = pd.read_csv('/tmp/mscapital/submission.csv')
assert len(sub)==647896
sub['prediction'] = pred[sub.sample_id.values] if sub.sample_id.max() < len(pred) else pred
sub.to_csv('/tmp/work/submission_gbm.csv', index=False)
print('submission written', sub.shape, 'pred std', float(np.std(pred)), flush=True)