# A: X2+X10 (118) main val. B: X2+X789+X10 (210) shift. C: X2+X10 (118) shift.
import numpy as np, gc, json, os
import lightgbm as lgb
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
n = len(y)
def stage(parts, trm, vam, tag):
    D = sum(np.load(f'/tmp/work/{p}_train.npy', mmap_mode='r').shape[1] for p in parts)
    ntr = int(trm.sum()); nva = int(vam.sum())
    Xtr = np.memmap(f'/tmp/work/{tag}_tr.f32', dtype=np.float32, mode='w+', shape=(ntr, D))
    Xv  = np.memmap(f'/tmp/work/{tag}_va.f32', dtype=np.float32, mode='w+', shape=(nva, D))
    col = 0
    for nm in parts:
        P = np.load(f'/tmp/work/{nm}_train.npy')
        d = P.shape[1]; ctr = 0; cva = 0
        for st in range(0, n, 100000):
            blk = P[st:st+100000].astype(np.float32, copy=False)
            m_ = ~np.isfinite(blk); blk[m_] = 0.0; del m_
            sb = trm[st:st+100000]; vb = vam[st:st+100000]
            k1 = int(sb.sum()); k2 = int(vb.sum())
            if k1: Xtr[ctr:ctr+k1, col:col+d] = blk[sb]
            if k2: Xv[cva:cva+k2, col:col+d] = blk[vb]
            ctr += k1; cva += k2
            del blk
        col += d; del P; gc.collect()
    Xtr.flush(); Xv.flush()
    return Xtr, Xv
def run(tag, parts, trm, vam, ref):
    Xtr, Xv = stage(parts, trm, vam, tag)
    yv = y[vam]
    params = dict(objective='regression', learning_rate=0.05, num_leaves=127,
                  min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
    preds=[]; iters=[]
    for seed in (7,42):
        p=dict(params); p['seed']=seed
        ds=lgb.Dataset(Xtr,y[trm]); dsv=lgb.Dataset(Xv,yv,reference=ds)
        m=lgb.train(p,ds,num_boost_round=3000,valid_sets=[dsv],callbacks=[lgb.early_stopping(100,verbose=False)])
        preds.append(m.predict(Xv,num_iteration=m.best_iteration)); iters.append(m.best_iteration)
        print(tag,'seed',seed,'best_iter',m.best_iteration,flush=True)
        del ds,dsv,m; gc.collect()
    pv=np.mean(preds,0); np.save(f'/tmp/work/{tag}_val.npy', pv)
    c = float(pv@yv/(np.linalg.norm(pv)*np.linalg.norm(yv)+1e-30))
    print(f'{tag} solo: {c:.6f} | ref {ref}', flush=True)
    json.dump(iters, open(f'/tmp/work/{tag}_iters.json','w'))
    del Xtr, Xv; gc.collect()
    os.remove(f'/tmp/work/{tag}_tr.f32'); os.remove(f'/tmp/work/{tag}_va.f32')
    return c
tr60 = month<=60; va61 = month>=61
tr50 = month<=50; va51 = (month>=51)&(month<=60)
run('x2x17', ('X2','X17'), tr60, va61, 'X2 0.123587')
run('xall17', ('X2','X7','X8','X9','X12','X13','X16','X17'), tr60, va61, 'X789X12X13X16 0.132116')
run('xall17shift', ('X2','X7','X8','X9','X12','X13','X16','X17'), tr50, va51, 'xall16 shift 0.128442')
print('X17AB_DONE', flush=True)
