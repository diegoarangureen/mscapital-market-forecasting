# RealMLP champion recipe -> 455f + X23 (148) + X24 (41) -> 644f.
# Single-variable test vs champion val 0.148940 (298f). tr<=60/val 61-70, cos loss.
# Architecture adapted from public kernel yunsuxiaozi/rfmf-realmlp (all-numerical, RQ aux dropped -
# RQ target aux tested negative by UnseenAnchor). Runs on CPU or GPU.
import os, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = int(os.environ.get('SEED', '2026'))
N_ENS = int(os.environ.get('N_ENS', '16'))
EPOCHS = int(os.environ.get('EPOCHS', '9'))
LR = float(os.environ.get('LR', '1e-3'))
BS = int(os.environ.get('BS', '256'))
def _resolve(fname):
    from pathlib import Path
    import subprocess
    for cand in [Path('/kaggle/input')]:
        r = subprocess.run(['find', str(cand), '-name', fname, '-maxdepth', '6'], capture_output=True, text=True, timeout=90)
        ls = [l for l in r.stdout.strip().split('\n') if l.strip()]
        if ls: return str(Path(ls[0]).parent)
    raise FileNotFoundError(fname)

DATA = os.environ.get('DATA', '') or _resolve('full_train.npy')
XTRA = os.environ.get('XTRA', '') or _resolve('X21_train.npy')
XTRA2 = os.environ.get('XTRA2', '') or _resolve('X22_train.npy')
D0726 = os.environ.get('D0726', '/kaggle/input/rfmf-0726data')

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, 'n_ens', N_ENS, 'epochs', EPOCHS, flush=True)


# ---------- data ----------
X_all = np.load(f'{DATA}/full_train.npy')          # (1257637, 246) float32, NaN->0
if os.environ.get('USE_XTRA', '1') == '1':
    Xt = np.load(f'{XTRA}/X21_train.npy').astype(np.float32)
    assert Xt.shape[0] == X_all.shape[0], Xt.shape
    X_all = np.concatenate([X_all, np.nan_to_num(Xt)], axis=1)
    Xt2 = np.load(f'{XTRA2}/X22_train.npy').astype(np.float32)
    assert Xt2.shape[0] == X_all.shape[0], Xt2.shape
    X_all = np.concatenate([X_all, np.nan_to_num(Xt2)], axis=1)
    print('XTRA+XTRA2 attached ->', X_all.shape, flush=True)
# drop top adversarial-validation discriminators (spread/slope regime features)
DROP = [301, 267, 268, 299, 257]
X_all = np.delete(X_all, DROP, axis=1)
print('after adv drop ->', X_all.shape, flush=True)
# attach yunsuxiaozi 0726 public features (152 numeric cols)
import pandas as pd
from pathlib import Path
d0726 = None
for cand in [D0726, '/kaggle/input/rfmf-0726data', '/kaggle/input/kernels/yunsuxiaozi/rfmf-0726data']:
    if Path(cand).exists() and (Path(cand)/'train.csv').exists():
        d0726 = cand; break
if d0726 is None:
    import subprocess
    r = subprocess.run(['find','/kaggle/input','-name','train.csv','-maxdepth','5'], capture_output=True, text=True, timeout=60)
    for line in r.stdout.strip().split('\n'):
        if '0726' in line.lower(): d0726 = str(Path(line).parent); break
assert d0726, '0726 data not found'
t726 = pd.read_csv(f'{d0726}/train.csv').sort_values('sample_id').reset_index(drop=True)
assert (t726['sample_id'].values == np.arange(len(t726))).all() or len(t726)==len(X_all)
dropc = ['sample_id','target','label','month','id']
fc = [c for c in t726.columns if c not in dropc and str(t726[c].dtype) in ('float64','float32','int64','int32')]
X726 = np.nan_to_num(t726[fc].values.astype(np.float32))
assert X726.shape[0] == X_all.shape[0], X726.shape
X_all = np.concatenate([X_all, X726], axis=1)
# ---------- X23 order-flow stream aggregates (148) ----------
x23_dir = None
for cand in ['/kaggle/input/mscapital-build-x23', '/kaggle/input/kernels/diegoaranguren/mscapital-build-x23']:
    if Path(cand).exists() and (Path(cand)/'X23_train.npy').exists():
        x23_dir = cand; break
if x23_dir is None:
    import subprocess
    r = subprocess.run(['find','/kaggle/input','-name','X23_train.npy','-maxdepth','5'], capture_output=True, text=True, timeout=60)
    ls = [l for l in r.stdout.strip().split('\n') if l.strip()]
    if ls: x23_dir = str(Path(ls[0]).parent)
assert x23_dir, 'X23 not found'
X23 = np.load(f'{x23_dir}/X23_train.npy').astype(np.float32)
assert X23.shape[0] == X_all.shape[0]
X_all = np.concatenate([X_all, np.nan_to_num(X23)], axis=1)
del X23
print('+X23 ->', X_all.shape, flush=True)
# ---------- X24 absolute microstructure features (41) ----------
x24_dir = None
for cand in ['/kaggle/input/datasets/diegoaranguren/mscapital-x24', '/kaggle/input/mscapital-x24']:
    if Path(cand).exists() and (Path(cand)/'X24_train.npy').exists():
        x24_dir = cand; break
if x24_dir is None:
    import subprocess
    r = subprocess.run(['find','/kaggle/input','-name','X24_train.npy','-maxdepth','6'], capture_output=True, text=True, timeout=60)
    ls = [l for l in r.stdout.strip().split('\n') if l.strip()]
    if ls: x24_dir = str(Path(ls[0]).parent)
assert x24_dir, 'X24 not found'
X24 = np.load(f'{x24_dir}/X24_train.npy').astype(np.float32)
assert X24.shape[0] == X_all.shape[0]
X_all = np.concatenate([X_all, np.nan_to_num(X24)], axis=1)
del X24
print('+X24 ->', X_all.shape, flush=True)
print('0726 attached ->', X_all.shape, 'added', len(fc), flush=True)
y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32)
month = np.load(f'{DATA}/full_month.npy')
TR_MAX = int(os.environ.get('TR_MAX', '60'))
VA_LO = int(os.environ.get('VA_LO', '61'))
VA_HI = int(os.environ.get('VA_HI', '70'))
TAG = os.environ.get('TAG', 'x644')
tr = month <= TR_MAX; va = (month >= VA_LO) & (month <= VA_HI)
late = month >= 66
Xtr = X_all[tr]; ytr = y_all[tr]
Xva = X_all[va]; yva = y_all[va]
late_mask = late[va]
print('tr', Xtr.shape, 'va', Xva.shape, flush=True)

# ---------- robust scaling (fit on train only) ----------
med = np.median(Xtr, axis=0)
qd = np.quantile(Xtr, 0.75, axis=0) - np.quantile(Xtr, 0.25, axis=0)
z = qd == 0.0
qd[z] = 0.5 * (Xtr.max(axis=0)[z] - Xtr.min(axis=0)[z])
fac = 1.0 / (qd + 1e-30); fac[qd == 0.0] = 0.0
def scale(X):
    s = fac[None, :] * (X - med[None, :])
    return s / np.sqrt(1 + (s / 3) ** 2)
Xtr = scale(Xtr).astype(np.float32)
Xva = scale(Xva).astype(np.float32)

Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)
if device.type == 'cuda':
    Xtr_t = Xtr_t.to(device); ytr_t = ytr_t.to(device)
    Xva_t = Xva_t.to(device); yva_t = yva_t.to(device)
n_feat = Xtr.shape[1]

# ---------- model ----------
class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__(); self.scale = nn.Parameter(torch.ones(n_ens, n_features))
    def forward(self, x): return x * self.scale[None, :, :]

class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=1.0):
        super().__init__()
        self.n_ens = n_ens; self.n_features = n_features; self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)
    def forward(self, x):
        b = x.shape[0]
        periodic = torch.cos(2 * np.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        t = torch.einsum('bnfh,nfhd->bnfd', periodic, self.w2)
        t = self.act(t + self.b2.unsqueeze(0))
        return torch.cat([x.unsqueeze(-1), t], dim=-1).view(b, self.n_ens, -1)

class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None
    def forward(self, x):
        x = torch.einsum('bni,nio->bno', x, self.weight) / np.sqrt(self.in_features)
        return x + self.bias if self.bias is not None else x

class RealMLP(nn.Module):
    def __init__(self, n_numerical, n_ens=8):
        super().__init__()
        act = nn.GELU
        self.n_ens = n_ens
        self.num_embed = PBLDEmbedding(n_features=n_numerical, hidden_dim=24, out_dim=3,
                                       freq_scale=1.0, n_ens=n_ens)
        total_dim = n_numerical * 3
        self.dropout = nn.Dropout(0.01)
        self.shared = nn.Sequential(
            nn.LayerNorm(total_dim),
            ScalingLayer(n_ens, total_dim),
            NTPLinear(n_ens, total_dim, 512), act(), self.dropout,
            NTPLinear(n_ens, 512, 512), act(), self.dropout,
            NTPLinear(n_ens, 512, 128), act(), self.dropout,
        )
        self.reg_head = NTPLinear(n_ens, 128, 1)
        mask = torch.ones(n_ens, total_dim, dtype=torch.bool)
        for i in range(n_ens):
            mask[i, i::n_ens // 2] = False
        self.register_buffer('feature_mask', mask)
    def forward(self, x_num):
        x = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x = self.num_embed(x)
        x = x * self.feature_mask.unsqueeze(0).float()
        f = self.shared(x)
        return self.reg_head(f).squeeze(-1)   # (batch, n_ens)

class EMA:
    def __init__(self, model, decay=0.998):
        self.model = model; self.decay = decay
        self.ema_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    def update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ema_state[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    def apply(self):
        orig = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                orig[n] = p.data.clone().detach(); p.data.copy_(self.ema_state[n])
        return orig
    def restore(self, orig):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in orig: p.data.copy_(orig[n])

def flat_anneal(v, progress, flat_ratio=0.5):
    if progress < flat_ratio: return v
    return v * (1 - (progress - flat_ratio) / (1 - flat_ratio))

def cos_np(p, t):
    p = p - p.mean(); t = t - t.mean()
    return float((p * t).sum() / (np.linalg.norm(p) + 1e-8) / (np.linalg.norm(t) + 1e-8))

def loss_fn(y_pred, y_true, lambda_cos=1.0):
    ypf = y_pred.reshape(-1)
    ytf = y_true.unsqueeze(1).expand(-1, y_pred.shape[1]).reshape(-1)
    w = torch.where(torch.abs(ytf) > 0.001, 0.5, 1.0)
    mse = (w * (ypf - ytf) ** 2).mean()
    pc = ypf - ypf.mean(); tc = ytf - ytf.mean()
    cos = (pc * tc).sum() / (pc.norm() + 1e-8) / (tc.norm() + 1e-8)
    return mse + lambda_cos * (1 - cos)

model = RealMLP(n_feat, N_ENS).to(device)
print('params:', sum(p.numel() for p in model.parameters()), flush=True)

scale_p = [p for n, p in model.named_parameters() if 'scale' in n]
pbld_p = [p for n, p in model.named_parameters() if 'num_embed' in n]
bias_p = [p for n, p in model.named_parameters() if 'bias' in n]
_grouped = {id(q) for q in scale_p + pbld_p + bias_p}
rest = [p for n, p in model.named_parameters() if id(p) not in _grouped]
opt = torch.optim.AdamW([
    {'params': scale_p, 'lr': LR * 20.0, 'weight_decay': 1e-3},
    {'params': pbld_p,  'lr': LR * 0.093, 'weight_decay': 1e-2},
    {'params': rest,    'lr': LR, 'weight_decay': 1e-2},
    {'params': bias_p,  'lr': LR * 0.1, 'weight_decay': 5e-3},
], betas=(0.9, 0.98))
base_lrs = [g['lr'] for g in opt.param_groups]
ema = EMA(model, 0.998)

def evaluate():
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xva_t), 2048):
            preds.append(model(Xva_t[i:i+2048]).mean(dim=1).cpu())
    return torch.cat(preds).numpy()

best_cos = -1; best_state = None; best_pv = None; t0 = time.time()
steps_per_epoch = (len(ytr_t) + BS - 1) // BS
total_steps = steps_per_epoch * EPOCHS
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(ytr_t), device=Xtr_t.device)
    for i in range(0, len(ytr_t), BS):
        idx = perm[i:i+BS]
        step = ep * steps_per_epoch + i // BS
        progress = min(step / total_steps, 1.0)
        for g, bl in zip(opt.param_groups, base_lrs):
            g['lr'] = flat_anneal(bl, progress)
        by = ytr_t[idx] + torch.randn_like(ytr_t[idx]) * (0.005 * (1 - progress))
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t[idx]), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update()
    orig = ema.apply()
    pv = evaluate()
    ema.restore(orig)
    c_full = cos_np(pv, yva); c_late = cos_np(pv[late_mask], yva[late_mask])
    print(f'ep {ep+1}/{EPOCHS} val_cos {c_full:.6f} late_cos {c_late:.6f} elapsed {time.time()-t0:.0f}s', flush=True)
    if c_full > best_cos:
        best_cos = c_full
        best_state = {k: v.cpu().clone() for k, v in ema.ema_state.items()}
        best_pv = pv.copy()
        np.save(f'/kaggle/working/val_pred_{TAG}.npy', pv)

json.dump({'best_val_cos': best_cos, 'n_ens': N_ENS, 'epochs': EPOCHS, 'seed': SEED,
           'device': device.type}, open(f'/kaggle/working/metrics_{TAG}.json', 'w'))
torch.save(best_state, f'/kaggle/working/best_ema_{TAG}.pt')
print('DONE best val_cos', best_cos, flush=True)
