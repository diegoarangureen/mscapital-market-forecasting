# MLP eval: 121 base feats + X12 + X13 (185 total). Val split tr<=60/va>=61, seeds 0/1, early stop.
import numpy as np, gc, time, json
import pyarrow.feather as feather
W='/tmp/work'; NG=30
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
X2 = np.load(f'{W}/X2_train.npy'); d2 = X2.shape[1]
n = len(X2)
D = d2 + (NG-1) + NG + np.load(f'{W}/X6_train.npy', mmap_mode='r').shape[1] + 64
M = np.memmap(f'{W}/mlp1213.f32', dtype=np.float32, mode='w+', shape=(n, D))
M[:, :d2] = X2; del X2; gc.collect()
col = d2
fpR = np.load(f'{W}/fpR_train.npy'); fpR = np.maximum(fpR, 1e-9); np.log(fpR, out=fpR)
M[:, col:col+NG-1] = np.diff(fpR, axis=1); del fpR; gc.collect(); col += NG-1
fpV = np.load(f'{W}/fpV_train.npy'); np.log1p(fpV, out=fpV)
M[:, col:col+NG] = fpV; del fpV; gc.collect(); col += NG
X6 = np.load(f'{W}/X6_train.npy'); d6 = X6.shape[1]
M[:, col:col+d6] = X6; del X6; gc.collect(); col += d6
for nm in ('X12','X13'):
    P = np.load(f'{W}/{nm}_train.npy')
    M[:, col:col+P.shape[1]] = P; col += P.shape[1]; del P; gc.collect()
assert col == D
np.nan_to_num(M, copy=False); np.clip(M, -1e6, 1e6, out=M)
s1=np.zeros(D); s2=np.zeros(D); B=100000
trm = month<=60
for st in range(0, n, B):
    blk=M[st:st+B]; s1+=blk.sum(0); s2+=(blk**2).sum(0)
mu=s1/n; sd=np.sqrt(np.maximum(s2/n-mu**2,0))+1e-6
for st in range(0, n, B):
    blk=M[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
print('matrix ready', D, flush=True)
import torch, torch.nn as nn
torch.set_num_threads(2)
yt = torch.from_numpy((y*1000.0).astype(np.float32))
tri = np.where(trm)[0]; vai = np.where((month>=61))[0]
yv = y[month>=61]; late = month[month>=61]>=66
def run(seed, max_ep=12):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs=8192; best=-1; bestpv=None; bad=0
    ytt = yt[tri]
    for ep in range(max_ep):
        perm = np.random.permutation(len(tri)); net.train()
        for i in range(0,len(tri),bs):
            bidx = tri[perm[i:i+bs]]
            xb = torch.from_numpy(np.asarray(M[bidx]))
            loss = ((net(xb).squeeze(-1)-ytt[perm[i:i+bs]])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = np.empty(len(vai))
            for i in range(0, len(vai), 65536):
                xb = torch.from_numpy(np.asarray(M[vai[i:i+65536]]))
                pv[i:i+65536] = net(xb).squeeze(-1).numpy()
        c = cos(pv, yv*1000.0)
        print(f'seed {seed} ep {ep} val cos {c:.6f}', flush=True)
        if c > best: best=c; bestpv=pv.copy(); bad=0
        else:
            bad+=1
            if bad>=4: break
    return bestpv, best
preds=[]
for seed in (0,1):
    pv, b = run(seed); preds.append(pv)
    print('seed', seed, 'best', round(b,6), flush=True)
pv = np.mean(preds,0); np.save(f'{W}/mlp1213_val.npy', pv)
u = unit(pv)
print(f'MLP+X12X13 solo: full {cos(pv,yv*1000.0):.6f} late {cos(u[late],(yv*1000.0)[late]):.6f}', flush=True)
ml0 = unit(np.load(f'{W}/mlp_seedavg_val.npy'))
g = unit(np.load(f'{W}/x7891213_val.npy'))
b1 = unit(0.5*g+0.5*ml0); b2 = unit(0.5*g+0.5*u); b3 = unit(0.4*g+0.3*ml0+0.3*u)
print(f'blend gbm+oldMLP: {cos(b1,yv):.6f}/{cos(b1[late],yv[late]):.6f}', flush=True)
print(f'blend gbm+newMLP: {cos(b2,yv):.6f}/{cos(b2[late],yv[late]):.6f}', flush=True)
print(f'blend 3way 40/30/30: {cos(b3,yv):.6f}/{cos(b3[late],yv[late]):.6f}', flush=True)
print('MLP1213_DONE', flush=True)
