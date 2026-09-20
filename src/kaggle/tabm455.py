# TabM (bestwater kgpu-tabm-cos689 architecture, Apache 2.0, reimplemented) on OUR 455f.
# k=64 heads on shared MLP backbone (512 x3 blocks, BN+GELU+dropout 0.15), cosine loss on RAW target.
# Same 5-fold purge scheme as our kfold455. FOLD_IDX selects folds (0-based) for budget control.
# No XS features (LB-toxic for us). Per-fold standardization (safer than his global <=62 scaler).
import os, json, time
import numpy as np
import torch
import torch.nn as nn

SEEDS = [int(s) for s in os.environ.get('SEEDS', '2026').split(',')]
K_HEADS = 64; HIDDEN = 512; N_BLOCKS = 3; DROPOUT = 0.15
EPOCHS = 60; BATCH_SIZE = 2048; LR = 1e-3; WD = 1e-5; PATIENCE = 15; GRAD_CLIP = 1.0
FOLD_IDX = set(int(x) for x in os.environ.get('FOLD_IDX', '3,4').split(','))
TAG = os.environ.get('TAG', 'tabm455')

def _resolve(fname):
    from pathlib import Path
    import subprocess
    r = subprocess.run(['find', '/kaggle/input', '-name', fname, '-maxdepth', '6'],
                       capture_output=True, text=True, timeout=90)
    ls = [l for l in r.stdout.strip().split('\n') if l.strip()]
    if ls: return str(Path(ls[0]).parent)
    raise FileNotFoundError(fname)

DATA = _resolve('full_train.npy'); XTRA = _resolve('X21_train.npy'); XTRA2 = _resolve('X22_train.npy')
X_all = np.load(f'{DATA}/full_train.npy')
X1 = np.load(f'{XTRA}/X21_train.npy').astype(np.float32)
X2 = np.load(f'{XTRA2}/X22_train.npy').astype(np.float32)
X_all = np.concatenate([X_all, np.nan_to_num(X1), np.nan_to_num(X2)], axis=1)
DROP = [301, 267, 268, 299, 257]
X_all = np.delete(X_all, DROP, axis=1)
del X1, X2
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
t726 = pd.read_csv(f'{d0726}/train.csv').sort_values('sample_id').reset_index(drop=True)
dropc = ['sample_id','target','label','month','id']
fc = [c for c in t726.columns if c not in dropc and str(t726[c].dtype) in ('float64','float32','int64','int32')]
X726 = np.nan_to_num(t726[fc].values.astype(np.float32))
X_all = np.concatenate([X_all, X726], axis=1); del X726, t726
y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32)
month = np.load(f'{DATA}/full_month.npy')
print('455f ->', X_all.shape, flush=True)

FOLDS = [
    (month >= 40) & (month <= 44),
    (month >= 50) & (month <= 54),
    (month >= 55) & (month <= 59),
    (month >= 60) & (month <= 64),
    (month >= 65) & (month <= 70),
]
TRAIN_MASKS = [month<=37, month<=47, month<=52, month<=57, month<=62]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count() if DEVICE.type == 'cuda' else 0
print('device:', DEVICE, 'gpus:', n_gpu, flush=True)

class TabM(nn.Module):
    def __init__(self, n_feat, k=64, hidden=512, n_blocks=3, dropout=0.15):
        super().__init__()
        layers = [nn.Linear(n_feat, hidden), nn.GELU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout)]
        for _ in range(n_blocks):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout)])
        layers.extend([nn.Linear(hidden, hidden//2), nn.GELU(), nn.BatchNorm1d(hidden//2), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden//2, 1) for _ in range(k)])
        for i, head in enumerate(self.heads):
            nn.init.xavier_uniform_(head.weight, gain=0.1 + 0.01*i)
            nn.init.zeros_(head.bias)
    def forward(self, x):
        h = self.backbone(x)
        return torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1).mean(dim=1)

def cos_np(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def cosine_loss(pred, target):
    p = pred - pred.mean()
    t = target - target.mean()
    return 1.0 - (p * t).sum() / (torch.norm(p) * torch.norm(t) + 1e-8)

n_feat = X_all.shape[1]
oof = np.full(len(X_all), np.nan, dtype=np.float32)
fold_report = []
t_all = time.time()

for seed in SEEDS:
    for fi, (tr_mask, va_mask) in enumerate(zip(TRAIN_MASKS, FOLDS)):
        if fi not in FOLD_IDX: continue
        torch.manual_seed(seed); np.random.seed(seed)
        mu = X_all[tr_mask].mean(axis=0); sd = X_all[tr_mask].std(axis=0) + 1e-8
        Xtr = ((X_all[tr_mask] - mu) / sd).astype(np.float32)
        Xva = ((X_all[va_mask] - mu) / sd).astype(np.float32)
        ytr = y_all[tr_mask]; yva = y_all[va_mask]
        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
        Xva_t = torch.tensor(Xva)
        if DEVICE.type == 'cuda':
            Xtr_t = Xtr_t.to(DEVICE); ytr_t = ytr_t.to(DEVICE); Xva_t = Xva_t.to(DEVICE)
        model = TabM(n_feat, K_HEADS, HIDDEN, N_BLOCKS, DROPOUT).to(DEVICE)
        if n_gpu > 1:
            model = nn.DataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best_cos = -1; best_state = None; pat = 0; t0 = time.time()
        n = len(ytr_t)
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, device=Xtr_t.device)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred = model(Xtr_t[idx])
                loss = cosine_loss(pred, ytr_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sched.step()
            model.eval(); preds = []
            with torch.no_grad():
                for i in range(0, len(Xva_t), 8192):
                    preds.append(model(Xva_t[i:i+8192]).cpu())
            pv = torch.cat(preds).numpy()
            c = cos_np(pv, yva)
            if c > best_cos:
                best_cos = c; pat = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_pv = pv.copy()
            else:
                pat += 1
                if pat >= PATIENCE: break
        print(f'seed {seed} fold {fi}: best raw cos {best_cos:.5f} ep {ep+1} ({time.time()-t0:.0f}s)', flush=True)
        fold_report.append({'seed': seed, 'fold': fi, 'val_cos': float(best_cos), 'epochs': ep+1, 's': round(time.time()-t0)})
        oof[va_mask] = best_pv
        np.save(f'/kaggle/working/oof_{TAG}_partial.npy', oof)
        json.dump({'folds': fold_report}, open(f'/kaggle/working/metrics_{TAG}_partial.json','w'))
        del model, opt, Xtr_t, ytr_t, Xva_t, Xtr, Xva
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

np.save(f'/kaggle/working/oof_{TAG}.npy', oof)
seen = ~np.isnan(oof)
res = {'folds': fold_report, 'elapsed_s': round(time.time()-t_all)}
for lo, hi, name in [(60,64,'60-64'),(65,70,'65-70'),(61,70,'61-70')]:
    m = (month>=lo)&(month<=hi)&seen
    if m.any(): res[f'oof_{name}'] = cos_np(oof[m], y_all[m])
json.dump(res, open(f'/kaggle/working/metrics_{TAG}.json','w'), indent=1)
print('RESULT', json.dumps(res), flush=True)
print('DONE', flush=True)
