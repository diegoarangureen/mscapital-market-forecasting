# MLP on 181-feat 60-grid matrix: train 0-60, val 61-70, early stop, 2 seeds + blend eval.
import numpy as np, gc, time, json
y = np.load('/tmp/work/mlp60_y.npy'); month = np.load('/tmp/work/mlp60_month.npy')
tr = month<=60; va = month>=61; late = month>=66
yv = (y[va]/1000.0)  # y stored x1000; metrics on raw y scale
X = np.load('/tmp/work/mlp60_Xstd.npy')
D = X.shape[1]
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
import torch, torch.nn as nn
torch.set_num_threads(2)
Xall = torch.from_numpy(X)
yt = torch.from_numpy(y.astype(np.float32))
idx_tr = torch.from_numpy(np.where(tr)[0]); idx_va = np.where(va)[0]
def run(seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best=-1; bestp=None; bad=0; be=-1; n=len(idx_tr); bs=8192
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
        print(f'  seed{seed} ep{ep} cos {c:.6f}', flush=True)
        if c>best: best=c; bestp=pv; bad=0; be=ep
        else:
            bad+=1
            if bad>=4: break
    return bestp, best, be
vp=[]; bes={}
for seed in (0,1):
    p,b,be=run(seed); vp.append(p); bes[str(seed)]=be
    print('mlp60 seed',seed,'best',round(b,6),'ep',be,flush=True)
json.dump(bes, open('/tmp/work/mlp60_best_epochs.json','w'))
m=np.mean(vp,0); np.save('/tmp/work/mlp60_seedavg_val.npy', m)
lv = late[va]
u = unit(m)
print('MLP60 seedavg: full', round(cos(m,yv),6), 'late', round(cos(u[lv],yv[lv]),6), flush=True)
g = unit(np.load('/tmp/work/x2base_seedavg_val.npy'))
m30 = unit(np.load('/tmp/work/mlp_seedavg_val.npy'))
print('corr mlp60-gbm', round(float(np.corrcoef(u,g)[0,1]),4), 'corr mlp60-mlp30', round(float(np.corrcoef(u,m30)[0,1]),4), flush=True)
for wm in (0.4,0.5,0.6):
    bl = unit((1-wm)*g + wm*u)
    print(f'blend gbm+mlp60 w={wm}: full {cos(bl,yv):.6f} late {cos(bl[lv],yv[lv]):.6f}', flush=True)
bl3 = unit(0.4*g + 0.3*m30 + 0.3*u)
print(f'3way gbm+mlp30+mlp60: full {cos(bl3,yv):.6f} late {cos(bl3[lv],yv[lv]):.6f}', flush=True)
print('MLP60_DONE', flush=True)
