# X30-TPU: Optiver-canon proprietary order-flow pack (parent green light Sep 24 06:51).
# 36 features: multi-level OFI (Cont-Kukanov-Stoikov), trade-sign autocorr lag1-3, RV family
# (rv60/rv600/Parkinson/semivariance/vol-of-vol), order arrival/cancel intensity, queue-imbalance
# dynamics, microprice/mid slopes, spread, book shape, full-window trade-ticker (vwap_dev/tvol/tcount).
# Built kernel-side by mscapital-x30-build v4 (normalized OFI, floored midbar, clipped tails).
# Screen: 5-fold single-session panel on seed 2026
# vs kfold seed2026 per-fold baselines f1 0.143251 / f2 0.149152 / f3 0.148980 / f4 0.155356 /
# f5 0.171549. Verdict rule (X-series): consistent shift > 0.002 = signal, else FLAT.
# v1 ERROR post-mortem: quantile block stripped but champion robust scaler (median/IQR soft-clamp)
# wrongly removed too -> raw features -> val cos halved (~0.08); plus NameError on `seen` after fold 5.
# v2 restores the exact baseline scaler. NO test predictions, NO submission output - pure OOF screen.
# XLA-safe pattern verbatim from
# the X28c port (per-epoch anneal, CPU permutation, BS 256, no recompile between folds).
# bestwater-style inference: 5 purged folds x N seeds, holdout+ES per fold, test pred = mean of fold-seed models.
# Folds: val 40-44/50-54/55-59/60-64/65-70, train <=37/47/52/57/62. Scaler fit per fold on its train.
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '32')
os.environ.setdefault('OMP_NUM_THREADS', '32')
os.environ.setdefault('MKL_NUM_THREADS', '32')
import os, math, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [int(s) for s in os.environ.get('SEEDS', '2026').split(',')]
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA = True
except ImportError:
    XLA = False
N_ENS = int(os.environ.get('N_ENS', '16'))
EPOCHS = int(os.environ.get('EPOCHS', '10'))
PATIENCE = int(os.environ.get('PATIENCE', '3'))
LR = float(os.environ.get('LR', '1e-3'))
BS = int(os.environ.get('BS', '256'))  # 256 = champion recipe; 1024+ only for later throughput experiments
TAG = os.environ.get('TAG', 'x30pairrep')
DATA = os.environ.get('DATA', '/kaggle/input/mscapital-matrices')
XTRA = os.environ.get('XTRA', '/kaggle/input/mscapital-x21')
XTRA2 = os.environ.get('XTRA2', '/kaggle/input/mscapital-x22')

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True
device = xm.xla_device() if XLA else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, 'seeds', SEEDS, 'XLA:', XLA, 'BS:', BS, flush=True)


# ---------- wait for dataset mounts (TPU VM mounts /kaggle/input async; X28b v1 crashed at 36s on missing full_train.npy) ----------
def wait_for(path, timeout=300):
    # TPU VM mounts are async AND the layout varies by session generation:
    # legacy /kaggle/input/datasets/<owner>/<slug>/ vs current /kaggle/input/<slug>/.
    # find-first (cheap), then poll.
    import subprocess
    t0 = time.time()
    while not os.path.exists(path):
        r = subprocess.run(['find', '/kaggle/input', '-name', '*' + os.path.basename(path)], capture_output=True, text=True, timeout=120)
        hits = [h for h in r.stdout.strip().split(chr(10)) if h]
        if hits:
            if hits[0] != path:
                print('mount relocate:', path, '->', hits[0], flush=True)
            return hits[0]
        if time.time() - t0 > timeout:
            raise FileNotFoundError(path)
        print('waiting for mount:', path, f'{time.time()-t0:.0f}s', flush=True)
        time.sleep(15)

def data_file(base, name):
    p = f'{base}/{name}'
    if os.path.exists(p):
        return p
    return wait_for(p)

# ---------- train features ----------
X_all = np.load(data_file(DATA, 'full_train.npy'))
X1 = np.load(data_file(XTRA, 'X21_train.npy')).astype(np.float32)
X2 = np.load(data_file(XTRA2, 'X22_train.npy')).astype(np.float32)
X_all = np.concatenate([X_all, np.nan_to_num(X1), np.nan_to_num(X2)], axis=1)
DROP = [301, 267, 268, 299, 257]
X_all = np.delete(X_all, DROP, axis=1)
del X1, X2
XTRA3 = os.environ.get('XTRA3', '/kaggle/input/mscapital-x30')
X3 = np.load(data_file(XTRA3, 'X30_train.npy')).astype(np.float32)
assert X3.shape[0] == X_all.shape[0], f'X30 rows {X3.shape[0]} != train rows {X_all.shape[0]}'
print('base 450f (pre-X726) ->', X_all.shape, flush=True)
y_all = np.load(data_file(DATA, 'full_y.npy')).astype(np.float32)
month = np.load(data_file(DATA, 'full_month.npy'))

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
X_all = np.concatenate([X_all, X726], axis=1); del X726, t726
Xb = X_all
Xx = np.concatenate([Xb, np.nan_to_num(X3)], axis=1); del X3
print('paired arms: base', Xb.shape, 'x30', Xx.shape, flush=True)

# ---------- model classes (verbatim from champion) ----------
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

def fit_scale(X):
    med = np.median(X, axis=0)
    qd = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
    z = qd == 0.0
    qd = qd.copy(); qd[z] = 0.5 * (X.max(axis=0)[z] - X.min(axis=0)[z])
    fac = 1.0 / (qd + 1e-30); fac[qd == 0.0] = 0.0
    return med, fac

def apply_scale(X, med, fac):
    s = fac[None, :] * (X - med[None, :])
    return (s / np.sqrt(1 + (s / 3) ** 2)).astype(np.float32)

FOLDS = [((month >= 40) & (month <= 44), month <= 37),
         ((month >= 50) & (month <= 54), month <= 47),
         ((month >= 55) & (month <= 59), month <= 52),
         ((month >= 60) & (month <= 64), month <= 57),
         ((month >= 65) & (month <= 70), month <= 62)]

oof = {'base': np.full(len(Xb), np.nan, dtype=np.float32), 'x30': np.full(len(Xb), np.nan, dtype=np.float32)}
oof_models = []   # per-model OOF vectors (NaN outside each model's val fold)
fold_report = []
n_models = 0
t_all = time.time()

FOLD_IDX = set(int(x) for x in os.environ.get('FOLD_IDX', '0,1,2,3,4').split(','))
for seed in SEEDS:
    for arm, Xm in (('base', Xb), ('x30', Xx)):
     for fi, (va_mask, tr_mask) in enumerate(FOLDS):
        if fi not in FOLD_IDX: continue
        set_seed(seed * 100 + fi)
        med, fac = fit_scale(Xm[tr_mask])
        Xtr = apply_scale(Xm[tr_mask], med, fac)
        Xva = apply_scale(Xm[va_mask], med, fac)
        ytr = y_all[tr_mask]; yva = y_all[va_mask]
        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
        Xva_t = torch.tensor(Xva)
        if device.type in ('cuda','xla'):
            Xtr_t = Xtr_t.to(device); ytr_t = ytr_t.to(device); Xva_t = Xva_t.to(device)
        n_feat = Xtr.shape[1]
        model = RealMLP(n_feat, N_ENS).to(device)
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
        best_cos = -1; best_state = None; pat = 0; t0 = time.time()
        steps_per_epoch = (len(ytr_t) + BS - 1) // BS
        total_steps = steps_per_epoch * EPOCHS
        for ep in range(EPOCHS):
            model.train()
            ep_progress = ep / max(EPOCHS - 1, 1)
            for g, bl in zip(opt.param_groups, base_lrs):
                g['lr'] = flat_anneal(bl, ep_progress)
            noise_std = 0.005 * (1 - ep_progress)  # per-EPOCH scalar: per-step constants = XLA recompile bomb
            perm_np = np.random.permutation(len(ytr_t))  # CPU: device-tensor slicing bakes offsets as graph constants
            for i in range(0, len(ytr_t), BS):
                idx = torch.from_numpy(perm_np[i:i+BS]).to(Xtr_t.device)
                by = ytr_t[idx] + torch.randn_like(ytr_t[idx]) * noise_std
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[idx]), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if XLA:
                    xm.optimizer_step(opt, barrier=False); xm.mark_step()
                else:
                    opt.step()
                ema.update()
            orig = ema.apply()
            model.eval(); preds = []
            with torch.no_grad():
                for i in range(0, len(Xva_t), 2048):
                    preds.append(model(Xva_t[i:i+2048]).mean(dim=1).cpu())
            pv = torch.cat(preds).numpy()
            ema.restore(orig)
            c = cos_np(pv, yva)
            if c > best_cos:
                best_cos = c; pat = 0
                best_state = {k: v.cpu().clone() for k, v in ema.ema_state.items()}
            else:
                pat += 1
                if pat >= PATIENCE: break
        print(f'{arm} seed {seed} fold {fi+1}: best val cos {best_cos:.6f} ({time.time()-t0:.0f}s)', flush=True)
        fold_report.append({'arm': arm, 'seed': seed, 'fold': fi+1, 'val_cos': float(best_cos), 's': round(time.time()-t0)})
        # predict with best EMA state
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xva_t), 4096):
                vp.append(model(Xva_t[i:i+4096]).mean(dim=1).cpu())
            _vp = torch.cat(vp).numpy()
            oof[arm][va_mask] = _vp
            _om = np.full(len(Xb), np.nan, dtype=np.float32); _om[va_mask] = _vp
            oof_models.append(_om)
            n_models += 1

        # incremental save: partial results survive a session timeout/cancel
        np.save(f'/kaggle/working/oof_models_{TAG}_partial.npy', np.stack(oof_models))
        _m = (month >= 61) & (month <= 70)
        def _oc(a):
            c = oof[a]; seen = ~np.isnan(c); mm = _m & seen
            return cos_np(c[mm], y_all[mm]) if mm.any() else None
        json.dump({'n_models_done': n_models, 'folds': fold_report,
                   'oof_cos_61_70_base_partial': _oc('base'), 'oof_cos_61_70_x30_partial': _oc('x30')},
                  open(f'/kaggle/working/metrics_{TAG}_partial.json','w'), indent=1)
        del model, opt, ema, Xtr_t, ytr_t, Xva_t, Xtr, Xva
        if device.type == 'cuda': torch.cuda.empty_cache()
        if XLA: xm.mark_step()

print('models trained:', n_models, 'total time', round(time.time()-t_all), 's', flush=True)
np.save(f'/kaggle/working/oof_{TAG}.npy', oof)
np.save(f'/kaggle/working/oof_models_{TAG}.npy', np.stack(oof_models))

# metrics: OOF cos on months 61-70 (comparable to our champion val protocol)
m6170 = (month >= 61) & (month <= 70)
late66 = (month >= 66) & (month <= 70)
m4064 = (month >= 40) & (month <= 64)
res = {'n_models': n_models, 'folds': fold_report, 'elapsed_s': round(time.time()-t_all)}
for a in ('base','x30'):
    res[f'oof_cos_61_70_{a}'] = cos_np(oof[a][m6170], y_all[m6170])
    res[f'oof_cos_66_70_{a}'] = cos_np(oof[a][late66], y_all[late66])
    res[f'oof_cos_40_64_{a}'] = cos_np(oof[a][m4064], y_all[m4064])
json.dump(res, open(f'/kaggle/working/metrics_{TAG}.json','w'), indent=1)
print('OOF 61-70 base:', res['oof_cos_61_70_base'], 'x30:', res['oof_cos_61_70_x30'], flush=True)
print('DONE', flush=True)
