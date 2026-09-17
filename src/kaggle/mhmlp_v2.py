# Multi-head MLP (shared BN backbone + 64 averaged linear heads, cosine loss on raw target)
# on our 303f-advdrop + yunsuxiaozi 0726 (152) + XS cross-sectional 75 built from our 303 base.
# Ideas from bestwater's Apache-2.0 public kernel (kgpu-tabm-cos689-3seed, LB 0.142), reimplemented.
# Val protocol: tr<=60 -> val 61-70, early stop (patience 15) on val raw cos.
import os, json, time
import faulthandler; faulthandler.enable()
import numpy as np
import torch
import torch.nn as nn

SEED = int(os.environ.get('SEED', '2026'))
EPOCHS = int(os.environ.get('EPOCHS', '60'))
PATIENCE = int(os.environ.get('PATIENCE', '15'))
BS = int(os.environ.get('BS', '1024'))
LR = float(os.environ.get('LR', '1e-3'))
K_HEADS = int(os.environ.get('K_HEADS', '32'))
HID = int(os.environ.get('HID', '512'))
DROPOUT = float(os.environ.get('DROPOUT', '0.15'))
TAG = os.environ.get('TAG', 'mhmlp')
DATA = os.environ.get('DATA', '/kaggle/input/datasets/diegoaranguren/mscapital-matrices')
XTRA = os.environ.get('XTRA', '/kaggle/input/datasets/diegoaranguren/mscapital-x21')
XTRA2 = os.environ.get('XTRA2', '/kaggle/input/datasets/diegoaranguren/mscapital-x22')

import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

# ---------- base features (303f, advdrop) ----------
X_all = np.load(f'{DATA}/full_train.npy')
X1 = np.load(f'{XTRA}/X21_train.npy').astype(np.float32)
X2 = np.load(f'{XTRA2}/X22_train.npy').astype(np.float32)
X_all = np.concatenate([X_all, np.nan_to_num(X1), np.nan_to_num(X2)], axis=1)
DROP = [301, 267, 268, 299, 257]
X_all = np.delete(X_all, DROP, axis=1)
y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32)
month = np.load(f'{DATA}/full_month.npy')
print('base 298f ->', X_all.shape, flush=True)

# ---------- 0726 public features (152) ----------
import pandas as pd
from pathlib import Path
d0726 = None
for cand in ['/kaggle/input/rfmf-0726data', '/kaggle/input/kernels/yunsuxiaozi/rfmf-0726data']:
    if Path(cand).exists() and (Path(cand)/'train.csv').exists():
        d0726 = cand; break
if d0726 is None:
    import subprocess
    r = subprocess.run(['find','/kaggle/input','-name','train.csv','-maxdepth','5'], capture_output=True, text=True, timeout=60)
    for line in r.stdout.strip().split('\n'):
        if '0726' in line.lower(): d0726 = str(Path(line).parent); break
assert d0726, '0726 not found'
t726 = pd.read_csv(f'{d0726}/train.csv').sort_values('sample_id').reset_index(drop=True)
dropc = ['sample_id','target','label','month','id']
fc = [c for c in t726.columns if c not in dropc and str(t726[c].dtype) in ('float64','float32','int64','int32')]
X726 = np.nan_to_num(t726[fc].values.astype(np.float32))
assert X726.shape[0] == X_all.shape[0]
X_all = np.concatenate([X_all, X726], axis=1)
print('+0726 ->', X_all.shape, flush=True)

# ---------- XS cross-sectional features from our base 303 ----------
import lightgbm as lgb
from scipy.stats import rankdata
rng = np.random.default_rng(42)
sub = rng.choice(len(X_all), 200000, replace=False)
dt = lgb.Dataset(X_all[sub, :298], y_all[sub])
mt = lgb.train(dict(objective='regression', learning_rate=0.1, num_leaves=63, min_child_samples=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbose=-1, seed=42, n_jobs=2),
               dt, num_boost_round=100)
top30 = np.argsort(mt.feature_importance('gain'))[::-1][:30]; top15 = top30[:15]
del dt, mt, sub
print('lgb top30 selected', flush=True)
pct = np.zeros((len(X_all), 30), np.float32)
for i, f in enumerate(top30):
    for m in np.unique(month):
        mk = month == m
        pct[mk, i] = (rankdata(X_all[mk, f]) / mk.sum() - 0.5).astype(np.float32)
zs = np.zeros((len(X_all), 15), np.float32)
for i, f in enumerate(top15):
    for m in np.unique(month):
        mk = month == m
        v = X_all[mk, f]
        zs[mk, i] = ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)
def cross_rank(data, idx):
    o = np.argsort(data[:, idx], axis=1)
    r = np.empty_like(o, dtype=np.float32)
    cols = np.arange(len(idx), dtype=np.float32) / max(len(idx)-1, 1)
    for j in range(len(data)):
        r[j, o[j]] = cols
    return r - 0.5
cr = cross_rank(X_all[:, :298], top30)
X_all = np.concatenate([X_all, pct, zs, cr], axis=1)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
print('+XS75 ->', X_all.shape, flush=True)

# ---------- split + standardize (fit on tr) ----------
tr = month <= 60; va = (month >= 61) & (month <= 70)
late = month >= 66
mu = X_all[tr].mean(0); sd = X_all[tr].std(0) + 1e-8
X_all = ((X_all - mu) / sd).astype(np.float32)
Xtr = torch.tensor(X_all[tr]); ytr = torch.tensor(y_all[tr])
Xva = torch.tensor(X_all[va]); yva = y_all[va]
late_mask = late[va]
if device.type == 'cuda':
    Xtr = Xtr.to(device); ytr = ytr.to(device); Xva = Xva.to(device)
print('tr', Xtr.shape, 'va', Xva.shape, flush=True)

# ---------- model ----------
class MHMLP(nn.Module):
    def __init__(self, n_feat, k=64, hid=512, blocks=3, drop=0.15):
        super().__init__()
        L = [nn.Linear(n_feat, hid), nn.GELU(), nn.BatchNorm1d(hid), nn.Dropout(drop)]
        for _ in range(blocks):
            L += [nn.Linear(hid, hid), nn.GELU(), nn.BatchNorm1d(hid), nn.Dropout(drop)]
        L += [nn.Linear(hid, hid//2), nn.GELU(), nn.BatchNorm1d(hid//2), nn.Dropout(drop)]
        self.backbone = nn.Sequential(*L)
        self.heads = nn.ModuleList([nn.Linear(hid//2, 1) for _ in range(k)])
        for i, h in enumerate(self.heads):
            nn.init.xavier_uniform_(h.weight, gain=0.1 + 0.01*i); nn.init.zeros_(h.bias)
    def forward(self, x):
        h = self.backbone(x)
        return torch.stack([hh(h).squeeze(-1) for hh in self.heads], 1).mean(1)

def cos_np(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def cosine_loss(pred, target):
    p = pred - pred.mean(); t = target - target.mean()
    return 1.0 - (p*t).sum() / (torch.norm(p)*torch.norm(t) + 1e-8)

model = MHMLP(Xtr.shape[1], K_HEADS, HID, 3, DROPOUT).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

def evaluate():
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xva), 8192):
            preds.append(model(Xva[i:i+8192]).cpu())
    return torch.cat(preds).numpy()

best = -1; best_pv = None; pat = 0; t0 = time.time()
n_tr = len(Xtr); n_full = (n_tr // BS) * BS
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(n_tr, device=Xtr.device)
    for s in range(0, n_full, BS):
        idx = perm[s:s+BS]
        loss = cosine_loss(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    pv = evaluate()
    c_full = cos_np(pv, yva); c_late = cos_np(pv[late_mask], yva[late_mask])
    print(f'ep {ep+1} val_cos {c_full:.6f} late {c_late:.6f} elapsed {time.time()-t0:.0f}s', flush=True)
    if c_full > best:
        best = c_full; best_pv = pv.copy(); pat = 0
        np.save(f'/kaggle/working/val_pred_{TAG}.npy', pv)
    else:
        pat += 1
        if pat >= PATIENCE: break
json.dump({'best_val_cos': best, 'k_heads': K_HEADS, 'hid': HID, 'seed': SEED},
          open(f'/kaggle/working/metrics_{TAG}.json','w'))
print('DONE best val_cos', best, flush=True)
