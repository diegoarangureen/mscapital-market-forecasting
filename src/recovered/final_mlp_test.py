# test preds + blend + CSV, standalone light process
import numpy as np, gc, time, json
W='/tmp/work'; NG=30
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
import torch, torch.nn as nn
torch.set_num_threads(2)
D = 58 + (NG-1) + NG + 4
def build(x2path, fpRpath, fpVpath, x6path):
    X2 = np.load(x2path); d2 = X2.shape[1]
    X6 = np.load(x6path); d6 = X6.shape[1]
    M = np.empty((len(X2), D), np.float32)
    M[:, :d2] = X2; del X2; gc.collect()
    fpR = np.load(fpRpath); fpR = np.maximum(fpR, 1e-9); np.log(fpR, out=fpR)
    M[:, d2:d2+NG-1] = np.diff(fpR, axis=1); del fpR; gc.collect()
    fpV = np.load(fpVpath); np.log1p(fpV, out=fpV)
    M[:, d2+NG-1:d2+2*NG-1] = fpV; del fpV; gc.collect()
    M[:, -d6:] = X6; del X6; gc.collect()
    np.nan_to_num(M, copy=False); np.clip(M, -1e6, 1e6, out=M)
    return M
mu = np.load(f'{W}/mlp_refit_mu.npy'); sd = np.load(f'{W}/mlp_refit_sd.npy')
Mt = build(f'{W}/X2_test.npy', f'{W}/fpR_test.npy', f'{W}/fpV_test.npy', f'{W}/X6_test.npy')
B=100000
for st in range(0, len(Mt), B):
    blk=Mt[st:st+B]; blk-=mu; blk/=sd; np.clip(blk,-10,10,out=blk)
XteT = torch.from_numpy(Mt)
def mknet():
    return nn.Sequential(nn.Linear(D,256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
mlp_test = np.zeros(len(Mt))
for seed in (0,1,2):
    net = mknet(); net.load_state_dict(torch.load(f'{W}/mlp_refit_seed{seed}.pt')); net.eval()
    with torch.no_grad():
        pv = np.empty(len(Mt))
        for i in range(0, len(Mt), 65536):
            pv[i:i+65536] = net(XteT[i:i+65536]).squeeze(-1).numpy()
    mlp_test += pv/3
    print(f'test preds seed {seed} done', flush=True)
np.save(f'{W}/refit_mlp_test.npy', mlp_test)
del Mt, XteT; gc.collect()
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); del lab; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
gbm_test = np.load(f'{W}/refit_gbm_test.npy'); gbm_train = np.load(f'{W}/refit_gbm_train.npy')
mlp_train = np.load(f'{W}/refit_mlp_train.npy')
nodata = np.load(f'{W}/test_nodata.npy')
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
