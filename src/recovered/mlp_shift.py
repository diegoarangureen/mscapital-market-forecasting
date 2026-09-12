# MLP on shifted split (train 0-50, val 51-60) for blend robustness
import numpy as np, gc, time
import pyarrow.feather as feather
import torch, torch.nn as nn
torch.set_num_threads(2)
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
y = lab.target.values.astype(np.float64)*1000.0; month = lab.month.values
tr = month<=50; va = (month>=51)&(month<=60)
yv = lab.target.values[va]
n_total = len(y)
X2 = np.load('/tmp/work/X2_train.npy'); d2 = X2.shape[1]
X6 = np.load('/tmp/work/X6_train.npy'); d6 = X6.shape[1]
fpR = np.load('/tmp/work/fpR_train.npy'); fpV = np.load('/tmp/work/fpV_train.npy')
NG = fpR.shape[1]
D = d2 + (NG-1) + NG + d6
X = np.empty((n_total, D), np.float32)
X[:, :d2] = X2; del X2; gc.collect()
fpR = np.maximum(fpR, 1e-9); np.log(fpR, out=fpR)
X[:, d2:d2+NG-1] = np.diff(fpR, axis=1); del fpR; gc.collect()
np.log1p(fpV, out=fpV)
X[:, d2+NG-1:d2+2*NG-1] = fpV; del fpV; gc.collect()
X[:, -d6:] = X6; del X6; gc.collect()
np.nan_to_num(X, copy=False); np.clip(X, -1e6, 1e6, out=X)
s1 = np.zeros(D); s2 = np.zeros(D); ntr = 0
B = 100000
for st in range(0, len(X), B):
    blk = X[st:st+B]; m = tr[st:st+B]; bm = blk[m]
    s1 += bm.sum(0); s2 += (bm**2).sum(0); ntr += m.sum()
mu = s1/ntr; sd = np.sqrt(np.maximum(s2/ntr-mu**2,0))+1e-6
for st in range(0, len(X), B):
    blk = X[st:st+B]; blk -= mu; blk /= sd; np.clip(blk,-10,10,out=blk)
print('standardized', flush=True)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def run(seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xall = torch.from_numpy(X)
    idx_tr = torch.from_numpy(np.where(tr)[0]); idx_va = np.where(va)[0]
    yt = torch.from_numpy(y.astype(np.float32))
    best=-1; bestp=None; bad=0; n=len(idx_tr); bs=8192
    for ep in range(30):
        perm = torch.randperm(n); net.train()
        for i in range(0,n,bs):
            idx = idx_tr[perm[i:i+bs]]
            loss = ((net(Xall[idx]).squeeze(-1)-yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        pv = np.empty(len(idx_va))
        with torch.no_grad():
            for i in range(0,len(idx_va),65536):
                j = torch.from_numpy(idx_va[i:i+65536])
                pv[i:i+65536] = net(Xall[j]).squeeze(-1).numpy()
        c = cos(pv, yv)
        if c>best: best=c; bestp=pv; bad=0
        else:
            bad+=1
            if bad>=4: break
    return bestp, best
vp=[]
for seed in (0,1,2):
    p,b=run(seed); vp.append(p); print('mlp seed',seed,'best',round(b,6),flush=True)
m=np.mean(vp,0)
u=(m-m.mean())/m.std()
print('SHIFT MLP seedavg:', round(cos(m,yv),6), 'centered:', round(cos(u,yv),6), flush=True)
g = np.load('/tmp/work/shift_gbm_val.npy'); gu=(g-g.mean())/g.std()
bl = 0.5*gu+0.5*u
print('SHIFT BLEND 50/50:', round(cos(bl,yv),6), flush=True)
print('SHIFT_DONE', flush=True)