# Phase B: MLP refit 0-70 + test preds + 50/50 blend + submission CSV. 2GB-RAM disciplined.
import numpy as np, gc, time, json
import pyarrow.feather as feather
W='/tmp/work'; NG=30
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)
del lab; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)

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

M, D = build(f'{W}/X2_train.npy', f'{W}/fpR_train.npy', f'{W}/fpV_train.npy', f'{W}/X6_train.npy')
s1=np.zeros(D); s2=np.zeros(D); B=100000
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
    t0=time.time(); net = run_refit(seed, ep); net.eval(); nets.append(net)
    with torch.no_grad():
        pv = np.empty(len(M))
        for i in range(0, len(M), 65536):
            pv[i:i+65536] = net(Xall[i:i+65536]).squeeze(-1).numpy()
    mlp_train += pv/3
    print(f'mlp seed {seed} refit {ep} epochs ({round(time.time()-t0)}s), train cos {cos(pv,y*1000.0):.6f}', flush=True)
np.save(f'{W}/refit_mlp_train.npy', mlp_train)
del M, Xall, yt; gc.collect()
for seed, net in zip((0,1,2), nets):
    torch.save(net.state_dict(), f'{W}/mlp_refit_seed{seed}.pt')
print('TRAIN_PART_DONE', flush=True)
