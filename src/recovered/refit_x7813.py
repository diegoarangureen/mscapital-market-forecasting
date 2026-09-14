# Refit GBM X2+X7+X8+X9 on 0-70, 3 seeds -> test + train preds.
import numpy as np, gc, sys, json
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)
del lab; gc.collect()
which = sys.argv[1]
X = None
if which in ('trainmodels','trainpred'):
    p0 = np.load('/tmp/work/X2_train.npy'); n, d0 = p0.shape
    X = np.empty((n,214), np.float32); X[:,:d0]=p0; del p0; gc.collect()
    col=d0
    for nm in ('X7','X8','X9','X12','X13'):
        P = np.load(f'/tmp/work/{nm}_train.npy')
        X[:,col:col+P.shape[1]]=P; col+=P.shape[1]; del P; gc.collect()
    for st in range(0,n,100000):
        blk = X[st:st+100000]; m_ = ~np.isfinite(blk); blk[m_]=0.0; del m_
    print('train matrix ready', flush=True)
params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
iters = [117, 113, 115]  # 1.1x of x7891213 [106,103,~104]
if which=='trainmodels':
    for seed,nr in zip((7,42,123), iters):
        p=dict(params); p['seed']=seed
        ds=lgb.Dataset(X,y); m=lgb.train(p,ds,num_boost_round=nr)
        m.save_model(f'/tmp/work/x7813_seed{seed}.txt')
        print('model seed',seed,'saved',flush=True); del ds,m; gc.collect()
    print('MODELS_DONE', flush=True)
elif which=='test':
    d0 = 58
    p0t = np.load('/tmp/work/X2_test.npy'); nt = len(p0t)
    Xt = np.empty((nt,214), np.float32); Xt[:,:d0]=p0t; del p0t; gc.collect()
    col=d0
    for nm in ('X7','X8','X9','X12','X13'):
        P = np.load(f'/tmp/work/{nm}_test.npy')
        Xt[:,col:col+P.shape[1]]=P; col+=P.shape[1]; del P; gc.collect()
    for st in range(0,nt,100000):
        blk = Xt[st:st+100000]; m_ = ~np.isfinite(blk); blk[m_]=0.0; del m_
    print('test matrix ready', flush=True)
    preds=[]
    for seed in (7,42,123):
        m = lgb.Booster(model_file=f'/tmp/work/x7813_seed{seed}.txt')
        out = np.empty(nt)
        for st in range(0,nt,200000): out[st:st+200000]=m.predict(Xt[st:st+200000])
        preds.append(out); print('seed',seed,'predicted',flush=True); del m,out; gc.collect()
    np.save('/tmp/work/refit_x7813_test.npy', np.mean(preds,0))
    print('X7813_REFIT_TEST_DONE', flush=True)
elif which=='trainpred':
    preds=[]
    for seed in (7,42,123):
        m = lgb.Booster(model_file=f'/tmp/work/x7813_seed{seed}.txt')
        out = np.empty(n)
        for st in range(0,n,200000): out[st:st+200000]=m.predict(X[st:st+200000])
        preds.append(out); print('seed',seed,'predicted',flush=True); del m,out; gc.collect()
    np.save('/tmp/work/refit_x7813_train.npy', np.mean(preds,0))
    print('X7813_REFIT_TRAIN_DONE', flush=True)
