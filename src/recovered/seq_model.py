# Small GRU over 60x6 bar sequences. Signal check on a train subset first.
import numpy as np, gc, time, json, sys
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64)*1000.0; month = lab.month.values
del lab; gc.collect()
tr = month<=60; va = month>=61; late = month>=66
yv = lab_target = None
import pyarrow.feather as feather2
lab2 = feather2.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
yv = lab2.target.values[va]; del lab2; gc.collect()
SUB = int(sys.argv[1]) if len(sys.argv)>1 else 400000   # train subset size
ns = 1257637; STEPS=60; CH=6
X = np.memmap('/tmp/work/seq_train.f16', dtype=np.float16, mode='r', shape=(ns,STEPS,CH))
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
# global channel stats from a sample of rows
rng = np.random.default_rng(0)
rows = X[rng.choice(ns, 60000, replace=False)].astype(np.float32)
mu = rows.reshape(-1,CH).mean(0); sd = rows.reshape(-1,CH).std(0)+1e-6
del rows; gc.collect()
np.save('/tmp/work/seq_mu.npy', mu); np.save('/tmp/work/seq_sd.npy', sd)
import torch, torch.nn as nn
torch.set_num_threads(2)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(CH, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        h,_ = self.gru(x)
        return self.head(h[:,-1]).squeeze(-1)
tr_idx = np.where(tr)[0]
rng.shuffle(tr_idx); tr_idx = tr_idx[:SUB]
va_idx = np.where(va)[0]
yt = torch.from_numpy(y.astype(np.float32))
def load_batch(idx):
    xb = X[np.sort(idx)].astype(np.float32)
    xb = (xb-mu)/sd
    np.clip(xb,-8,8,out=xb)
    return torch.from_numpy(xb)
def run(seed, epochs=8):
    torch.manual_seed(seed)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    best=-1; bestp=None; bad=0
    n=len(tr_idx); bs=4096
    for ep in range(epochs):
        perm = torch.randperm(n); net.train(); t0=time.time()
        for i in range(0,n,bs):
            idx = tr_idx[perm[i:i+bs].numpy()]
            xb = load_batch(idx)
            pred = net(xb)
            loss = ((pred-yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); pv=[]
        with torch.no_grad():
            for i in range(0,len(va_idx),32768):
                pv.append(net(load_batch(va_idx[i:i+32768])).numpy())
        pv=np.concatenate(pv)
        c = cos(pv, yv)
        print(f'  seed{seed} ep{ep} valcos {c:.6f} ({round(time.time()-t0)}s)', flush=True)
        if c>best: best=c; bestp=pv; bad=0
        else:
            bad+=1
            if bad>=3: break
    return bestp, best
vp=[]
for seed in (0,1):
    p,b=run(seed); vp.append(p); print('seq seed',seed,'best',round(b,6),flush=True)
avg=np.mean(vp,0); u=unit(avg); lv=late[va]
print('SEQ seedavg full:', round(cos(avg,yv),6), 'centered:', round(cos(u,yv),6), ' LATE:', round(cos(u[lv],yv[lv]),6), flush=True)
np.save('/tmp/work/seq_seedavg_val.npy', avg)
g=np.load('/tmp/work/x2base_seedavg_val.npy'); m=np.load('/tmp/work/mlp_seedavg_val.npy')
gu=unit(g); mu_=unit(m)
for w in (0.1,0.2,0.3):
    bl=(0.5-w/2)*gu+(0.5-w/2)*mu_+w*u
    print(f'gbm+mlp+seq w_seq={w}: full {cos(bl,yv):.6f} late {cos(bl[lv],yv[lv]):.6f}', flush=True)
print('SEQ_MODEL_DONE', flush=True)
