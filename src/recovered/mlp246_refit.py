# Refit MLP246 (pure GBM feats) on months 0-70. Epochs = 1.1x best [9,4,~6.5] -> [10,4,7]. Seeds 0/1/2.
import numpy as np, gc, time
import pyarrow.feather as feather
W='/tmp/work'
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); del lab; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
X2 = np.load(f'{W}/X2_train.npy'); d2 = X2.shape[1]; n = len(X2); D = 246
M = np.memmap(f'{W}/mlp246r.f32', dtype=np.float32, mode='w+', shape=(n, D))
M[:, :d2] = X2; del X2; gc.collect()
col = d2
for nm in ('X7','X8','X9','X12','X13','X16'):
    P = np.load(f'{W}/{nm}_train.npy')
    M[:, col:col+P.shape[1]] = P; col += P.shape[1]; del P; gc.collect()
s1=np.zeros(D); s2=np.zeros(D); B=100000
for st in range(0, n, B):
    blk=M[st:st+B]; np.nan_to_num(blk, copy=False); np.clip(blk, -1e6, 1e6, out=blk)
    s1+=blk.sum(0); s2+=(blk**2).sum(0)
mu=s1/n; sd=np.sqrt(np.maximum(s2/n-mu**2,0))+1e-6
for st in range(0, n, B):
    blk=M[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
M.flush(); print('train matrix ready', flush=True)
import torch, torch.nn as nn
torch.set_num_threads(2)
yt = (y*1000.0).astype(np.float32)
def run(seed, epochs):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs=8192
    for ep in range(epochs):
        perm = np.random.permutation(n); net.train()
        for i in range(0,n,bs):
            bidx = perm[i:i+bs]
            xb = torch.from_numpy(np.asarray(M[bidx]))
            yb = torch.from_numpy(yt[bidx])
            loss = ((net(xb).squeeze(-1)-yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net
nets=[]
mlp_train = np.zeros(n)
for seed, ep in zip((0,1,2), (10,4,7)):
    t0=time.time(); net = run(seed, ep); net.eval(); nets.append(net)
    with torch.no_grad():
        pv = np.empty(n)
        for i in range(0, n, 65536):
            pv[i:i+65536] = net(torch.from_numpy(np.asarray(M[i:i+65536]))).squeeze(-1).numpy()
    mlp_train += pv/3
    print(f'seed {seed} {ep} ep ({round(time.time()-t0)}s) train cos {cos(pv, yt):.6f}', flush=True)
np.save(f'{W}/refit_mlp246_train.npy', mlp_train)
del M, mlp_train; gc.collect()
import os; os.remove(f'{W}/mlp246r.f32')
# test matrix
X2t = np.load(f'{W}/X2_test.npy'); nt = len(X2t)
Mt = np.memmap(f'{W}/mlp246t.f32', dtype=np.float32, mode='w+', shape=(nt, D))
Mt[:, :d2] = X2t; del X2t; gc.collect()
col = d2
for nm in ('X7','X8','X9','X12','X13','X16'):
    P = np.load(f'{W}/{nm}_test.npy')
    Mt[:, col:col+P.shape[1]] = P; col += P.shape[1]; del P; gc.collect()
for st in range(0, nt, B):
    blk=Mt[st:st+B]; np.nan_to_num(blk, copy=False); np.clip(blk, -1e6, 1e6, out=blk)
    blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
Mt.flush(); print('test matrix ready', flush=True)
mlp_test = np.zeros(nt)
with torch.no_grad():
    for net in nets:
        pv = np.empty(nt)
        for i in range(0, nt, 65536):
            pv[i:i+65536] = net(torch.from_numpy(np.asarray(Mt[i:i+65536]))).squeeze(-1).numpy()
        mlp_test += pv/3
np.save(f'{W}/refit_mlp246_test.npy', mlp_test)
os.remove(f'{W}/mlp246t.f32')
print('MLP246_REFIT_DONE', flush=True)
