# MLP refit on months 0-70 + pseudo-labeled test rows (confident half, w=0.1).
# Pseudo labels: 50/50 unit blend of refit GBM + refit MLP test preds (rescaled to y*1000 units).
import numpy as np, gc, time, json
import pyarrow.feather as feather
W='/tmp/work'; NG=30
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)
del lab; gc.collect()
def build(x2path, fpRpath, fpVpath, x6path):
    X2 = np.load(x2path); d2 = X2.shape[1]
    X6 = np.load(x6path); d6 = X6.shape[1]
    D = d2 + (NG-1) + NG + d6
    M = np.empty((len(X2), D), np.float32)
    M[:, :d2] = X2; del X2; gc.collect()
    fpR = np.load(fpRpath); fpR = np.maximum(fpR, 1e-9); np.log(fpR, out=fpR)
    M[:, d2:d2+NG-1] = np.diff(fpR, axis=1); del fpR; gc.collect()
    fpV = np.load(fpVpath); np.log1p(fpV, out=fpV)
    M[:, d2+NG-1:d2+2*NG-1] = fpV; del fpV; gc.collect()
    M[:, -d6:] = X6; del X6; gc.collect()
    np.nan_to_num(M, copy=False); np.clip(M, -1e6, 1e6, out=M)
    return M, D
Mtr, D = build(f'{W}/X2_train.npy', f'{W}/fpR_train.npy', f'{W}/fpV_train.npy', f'{W}/X6_train.npy')
mu = np.load(f'{W}/mlp_refit_mu.npy'); sd = np.load(f'{W}/mlp_refit_sd.npy')
B=100000
for st in range(0, len(Mtr), B):
    blk=Mtr[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
print('train matrix standardized', flush=True)
# pseudo labels from blend
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
g = np.load(f'{W}/refit_gbm_test.npy'); m = np.load(f'{W}/refit_mlp_test.npy')
pl_unit = 0.5*unit(g) + 0.5*unit(m); del g, m; gc.collect()
# rescale to y*1000 units: match std of train y*1000
pl = pl_unit * (y.std()*1000.0)
conf = np.abs(pl_unit) >= np.median(np.abs(pl_unit))
print('confident test rows:', int(conf.sum()), flush=True)
Mte, _ = build(f'{W}/X2_test.npy', f'{W}/fpR_test.npy', f'{W}/fpV_test.npy', f'{W}/X6_test.npy')
Mte = Mte[conf]
for st in range(0, len(Mte), B):
    blk=Mte[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
print('test confident matrix standardized', flush=True)
n_tr = len(Mtr); n_cf = len(Mte)
ytr = (y*1000.0).astype(np.float32)
ycf = pl[conf].astype(np.float32)
del pl, y; gc.collect()
print('rows: train', n_tr, 'conf', n_cf, flush=True)
import torch, torch.nn as nn
torch.set_num_threads(2)
Xtr_t = torch.from_numpy(Mtr); Xcf_t = torch.from_numpy(Mte)
yt1 = torch.from_numpy(ytr); yt2 = torch.from_numpy(ycf)
bes = json.load(open(f'{W}/mlp_best_epochs.json'))
def run_refit(seed, epochs):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs1, bs2 = 7168, 1024
    steps_per_ep = n_tr // bs1
    for ep in range(epochs):
        net.train(); t0=time.time()
        perm1 = torch.randperm(n_tr); perm2 = torch.randperm(n_cf)
        for stp in range(steps_per_ep):
            i1 = perm1[stp*bs1:(stp+1)*bs1]
            i2 = perm2[(stp*bs2)%n_cf:(stp*bs2)%n_cf+bs2]
            if len(i2)<bs2:
                i2 = torch.cat([i2, perm2[:bs2-len(i2)]])
            r1 = net(Xtr_t[i1]).squeeze(-1)-yt1[i1]
            r2 = net(Xcf_t[i2]).squeeze(-1)-yt2[i2]
            loss = (r1*r1).mean() + 0.1*(r2*r2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f'  seed{seed} ep{ep} ({round(time.time()-t0)}s)', flush=True)
    return net
for seed in (0,1,2):
    ep = max(1, int(round(bes[str(seed)]*1.1)))
    t0=time.time(); net = run_refit(seed, ep); net.eval()
    torch.save(net.state_dict(), f'{W}/mlp_pseudo_seed{seed}.pt')
    print(f'pseudo-mlp seed {seed} refit {ep} epochs ({round(time.time()-t0)}s)', flush=True)
    del net; gc.collect()
print('MLP_PSEUDO_DONE', flush=True)
