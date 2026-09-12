import numpy as np, gc, time
import pyarrow.feather as feather
import torch, torch.nn as nn

torch.set_num_threads(2)
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
y = lab.target.values.astype(np.float64)*1000.0; month = lab.month.values
tr = month<=60; va = month>=61
yv = lab.target.values[va]

X2 = np.load('/tmp/work/X2_train.npy')
fpR = np.load('/tmp/work/fpR_train.npy'); fpV = np.load('/tmp/work/fpV_train.npy')
X6 = np.load('/tmp/work/X6_train.npy')
lr = np.diff(np.log(np.maximum(fpR,1e-9)), axis=1)
lv = np.log1p(fpV)
X = np.concatenate([X2, lr, lv, X6], axis=1).astype(np.float32)
del X2, fpR, fpV, X6, lr, lv; gc.collect()
np.nan_to_num(X, copy=False); np.clip(X, -1e6, 1e6, out=X)
mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6
X = (X-mu)/sd; np.clip(X, -10, 10, out=X)
print('X', X.shape, flush=True)

def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

def run(seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1],256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr], dtype=torch.float32)
    Xv = torch.tensor(X[va])
    best=-1; bestp=None; bad=0
    n=len(Xt); bs=8192
    for ep in range(30):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            pred = net(Xt[idx]).squeeze(-1)
            loss = ((pred-yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv).squeeze(-1).numpy().astype(np.float64)
        c = cos(pv, yv)
        if c>best: best=c; bestp=pv; bad=0
        else:
            bad+=1
            if bad>=4: break
        print(f'  seed{seed} ep{ep} valcos {c:.6f}', flush=True)
    return bestp, best

vp=[]
for seed in (0,1,2):
    p,b = run(seed); vp.append(p); print('seed',seed,'best',round(b,6),flush=True)
avg=np.mean(vp,0)
u=avg-avg.mean(); u=u/u.std()
print('MLP seedavg:', round(cos(avg,yv),6), 'centered:', round(cos(u,yv),6), flush=True)
np.save('/tmp/work/mlp_seedavg_val.npy', avg)
# blend with GBM baseline
g = np.load('/tmp/work/x2base_seedavg_val.npy')
gu = (g-g.mean())/g.std()
for w in (0.2,0.3,0.4,0.5):
    bl = (1-w)*gu + w*u
    print(f'blend w_mlp={w}: {cos(bl,yv):.6f}', flush=True)
print('MLP_DONE', flush=True)