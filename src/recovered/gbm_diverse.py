# Diverse-config GBM: different hyperparams for ensemble diversity (vs same-config seed avg).
import numpy as np, gc, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
X = np.load('/tmp/work/X2_train.npy')
tr = month<=60; va = month>=61; late = month>=66
yv = y[va]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
base = dict(objective='regression', learning_rate=0.05, num_leaves=127,
            min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
            bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
variants = {
 'B_deep': dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.8, lambda_l2=1.0),
 'C_shallow': dict(learning_rate=0.07, num_leaves=63, min_data_in_leaf=800, feature_fraction=0.9, bagging_fraction=0.6, lambda_l2=2.0),
}
res = {}
allpv = {}
for tag, over in variants.items():
    p0 = dict(base); p0.update(over)
    preds = []
    for seed in (7,42):
        p = dict(p0); p['seed']=seed
        ds = lgb.Dataset(X[tr], y[tr])
        dsv = lgb.Dataset(X[va], yv, reference=ds)
        m = lgb.train(p, ds, num_boost_round=4000, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(m.predict(X[va], num_iteration=m.best_iteration))
        print(tag, 'seed', seed, 'best_iter', m.best_iteration, flush=True)
        del ds, dsv, m; gc.collect()
    pv = np.mean(preds,0); allpv[tag]=pv
    res[tag] = dict(full=round(cos(pv,yv),6), late=round(cos(unit(pv)[late[va]], yv[late[va]]),6))
    print(f'{tag} solo: full {res[tag]["full"]:.6f} late {res[tag]["late"]:.6f}', flush=True)
    del preds; gc.collect()
np.save('/tmp/work/gbmB_val.npy', allpv['B_deep']); np.save('/tmp/work/gbmC_val.npy', allpv['C_shallow'])
g = unit(np.load('x2base_seedavg_val.npy')); ml = unit(np.load('mlp_seedavg_val.npy'))
lv = late[va]
for tag in variants:
    v = unit(allpv[tag])
    print('corr base-', tag, round(float(np.corrcoef(g, v)[0,1]),4), flush=True)
    for wv in (0.15, 0.25):
        bl = unit(0.5*(1-wv)*g + 0.5*(1-wv)*ml + wv*v)
        print(f'blend+{tag} w={wv}: full {cos(bl,yv):.6f} late {cos(bl[lv],yv[lv]):.6f}', flush=True)
json.dump(res, open('/tmp/work/gbm_diverse_results.json','w'), indent=1)
print('DIVERSE_DONE', flush=True)
