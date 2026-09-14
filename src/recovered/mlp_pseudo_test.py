# Predict test with the 3 pseudo-MLP nets -> mlp_pseudo_test.npy, then build submission v3.
import numpy as np, gc, time
W='/tmp/work'; NG=30
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
Mte, D = build(f'{W}/X2_test.npy', f'{W}/fpR_test.npy', f'{W}/fpV_test.npy', f'{W}/X6_test.npy')
mu = np.load(f'{W}/mlp_refit_mu.npy'); sd = np.load(f'{W}/mlp_refit_sd.npy')
for st in range(0, len(Mte), 100000):
    blk=Mte[st:st+100000]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
print('test matrix ready', flush=True)
import torch, torch.nn as nn
torch.set_num_threads(2)
Xt = torch.from_numpy(Mte)
pred = np.zeros(len(Mte), np.float64)
for seed in (0,1,2):
    net = nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    net.load_state_dict(torch.load(f'{W}/mlp_pseudo_seed{seed}.pt'))
    net.eval()
    with torch.no_grad():
        pv = np.empty(len(Mte))
        for i in range(0, len(Mte), 65536):
            pv[i:i+65536] = net(Xt[i:i+65536]).squeeze(-1).numpy()
    pred += pv/3
    print('seed', seed, 'predicted', flush=True)
    del net; gc.collect()
np.save(f'{W}/mlp_pseudo_test.npy', pred)
print('MLP_PSEUDO_TEST_DONE', flush=True)
