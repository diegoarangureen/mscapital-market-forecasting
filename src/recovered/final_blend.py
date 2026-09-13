# Final submission builder: GBM v4 refit + MLP refit on months 0-70, 50/50 unit blend.
# Memory-disciplined for 2GB box: phases sequenced, big arrays freed aggressively.
import numpy as np, gc, time, json
import pyarrow.feather as feather
import lightgbm as lgb

W='/tmp/work'
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)
del lab; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)

# ---------- Phase A: GBM refit (champion recipe) ----------
X = np.load(f'{W}/X2_train.npy'); keys=list(np.load(f'{W}/X2_train_keys.npy'))
Xte = np.load(f'{W}/X2_test.npy')
kin = {k:i for i,k in enumerate(keys)}
nodata = (Xte[:, kin['tx_n']]==0) & (Xte[:, kin['mk_nbars']]==0)
print('no-data test samples:', int(nodata.sum()), flush=True)
P0 = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
          feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
          num_threads=2, verbose=-1)
iters = json.load(open(f'{W}/x2base_iters.json'))
gbm_test = np.zeros(len(Xte)); gbm_train = np.zeros(len(X))
dall = lgb.Dataset(X, label=y, feature_name=keys, free_raw_data=False)
for seed, it in zip((7,42,123), iters):
    n_it = int(round(it*1.1)); t0=time.time()
    m = lgb.train(dict(P0, seed=seed), dall, num_boost_round=n_it)
    gbm_test += m.predict(Xte); gbm_train += m.predict(X)
    print(f'gbm seed {seed} refit {n_it} it ({round(time.time()-t0)}s)', flush=True)
    del m; gc.collect()
gbm_test/=3; gbm_train/=3
print('gbm train cos (all months, in-sample):', round(cos(gbm_train,y),6), flush=True)
np.save(f'{W}/refit_gbm_test.npy', gbm_test); np.save(f'{W}/refit_gbm_train.npy', gbm_train)
del dall, Xte; gc.collect()

# ---------- Phase B: MLP refit ----------
NG=30
def assemble(x2, fpR, fpV, x6, n):
    d2 = x2.shape[1]; d6 = x6.shape[1]
    D = d2 + (NG-1) + NG + d6
    M = np.empty((n, D), np.float32)
    M[:, :d2] = x2
    fpR = np.maximum(fpR, 1e-9); np.log(fpR, out=fpR)
    M[:, d2:d2+NG-1] = np.diff(fpR, axis=1); del fpR
    np.log1p(fpV, out=fpV)
    M[:, d2+NG-1:d2+2*NG-1] = fpV; del fpV
    M[:, -d6:] = x6; del x6
    np.nan_to_num(M, copy=False); np.clip(M, -1e6, 1e6, out=M)
    return M, D

fpR = np.load(f'{W}/fpR_train.npy'); fpV = np.load(f'{W}/fpV_train.npy'); X6 = np.load(f'{W}/X6_train.npy')
M, D = assemble(X, fpR, fpV, X6, len(X))
del X; gc.collect()
# standardize on ALL train months (refit): block pass
s1=np.zeros(D); s2=np.zeros(D)
B=100000
for st in range(0, len(M), B):
    blk=M[st:st+B]; s1+=blk.sum(0); s2+=(blk**2).sum(0)
mu=s1/len(M); sd=np.sqrt(np.maximum(s2/len(M)-mu**2,0))+1e-6
for st in range(0, len(M), B):
    blk=M[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
print('mlp train matrix standardized', flush=True)
np.save(f'{W}/mlp_refit_mu.npy', mu); np.save(f'{W}/mlp_refit_sd.npy', sd)

import torch, torch.nn as nn
torch.set_num_threads(2)
yt = torch.from_numpy((y*1000.0).astype(np.float32))
Xall = torch.from_numpy(M)
bes = json.load(open(f'{W}/mlp_best_epochs.json'))
def run_refit(seed, epochs):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n=len(Xall); bs=8192
    for ep in range(epochs):
        perm = torch.randperm(n); net.train()
        for i in range(0,n,bs):
            idx = perm[i:i+bs]
            loss = ((net(Xall[idx]).squeeze(-1)-yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net
nets=[]
mlp_train = np.zeros(len(M))
for seed in (0,1,2):
    ep = max(1, int(round(bes[str(seed)]*1.1)))
    t0=time.time(); net = run_refit(seed, ep); nets.append(net)
    net.eval()
    with torch.no_grad():
        pv = np.empty(len(M))
        for i in range(0, len(M), 65536):
            pv[i:i+65536] = net(Xall[i:i+65536]).squeeze(-1).numpy()
    mlp_train += pv/3
    print(f'mlp seed {seed} refit {ep} epochs ({round(time.time()-t0)}s), train cos {cos(pv,y*1000.0):.6f}', flush=True)
np.save(f'{W}/refit_mlp_train.npy', mlp_train)
del M, Xall, yt; gc.collect()

# test matrix
Xte = np.load(f'{W}/X2_test.npy')
fpRt = np.load(f'{W}/fpR_test.npy'); fpVt = np.load(f'{W}/fpV_test.npy'); X6t = np.load(f'{W}/X6_test.npy')
Mt, _ = assemble(Xte, fpRt, fpVt, X6t, len(Xte))
del Xte; gc.collect()
for st in range(0, len(Mt), B):
    blk=Mt[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
XteT = torch.from_numpy(Mt)
mlp_test = np.zeros(len(Mt))
with torch.no_grad():
    for net in nets:
        pv = np.empty(len(Mt))
        for i in range(0, len(Mt), 65536):
            pv[i:i+65536] = net(XteT[i:i+65536]).squeeze(-1).numpy()
        mlp_test += pv/3
np.save(f'{W}/refit_mlp_test.npy', mlp_test)
del Mt, XteT, nets; gc.collect()

# ---------- Blend + submission ----------
pred = 0.5*unit(gbm_test) + 0.5*unit(mlp_test)
ptr = 0.5*unit(gbm_train) + 0.5*unit(mlp_train)
print('blend train cos (all, in-sample):', round(cos(ptr,y),6), flush=True)
pred[nodata] = 0.0
lo, hi = np.quantile(ptr, [0.001, 0.999])
pred = np.clip(pred, lo, hi)
import pandas as pd
sub = pd.read_csv('/tmp/mscapital/submission.csv')
assert len(sub)==len(pred) and (sub.sample_id.values==np.arange(len(sub))).all()
sub['prediction'] = pred
sub.to_csv(f'{W}/submission_blend.csv', index=False)
print('submission_blend.csv written', sub.shape, 'pred std', float(np.std(pred)), flush=True)
print('FINAL_BLEND_DONE', flush=True)
