# X28b-TPU: champion (lambda_cos=1.0) + RQ-KMeans auxiliary target-code head
# (3 layers x 3 codes, lambda_rq=0.1, yunsu rfmf-realmlp config reimplemented) on the
# torch_xla TPU port. Screen: seed 2026 fold5 vs replication target 0.170090. Gate +0.004.
# XLA-compile-safety review (Sep 21): lazy XLA bakes python scalars/slice offsets as graph
# constants -> per-step variants = thousands of graphs = LLVM compile OOM (root cause of
# tpuval455b/c deaths). Fixes: LR AND label-noise anneal per-EPOCH (<=10 graph variants),
# batch permutation stays on CPU (index tensor is graph DATA, not constants), BS default
# back to champion 256 for recipe fidelity (first TPU run must be a replication).
# bestwater-style inference: 5 purged folds x N seeds, holdout+ES per fold, test pred = mean of fold-seed models.
# Folds: val 40-44/50-54/55-59/60-64/65-70, train <=37/47/52/57/62. Scaler fit per fold on its train.
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
TAG = os.environ.get('TAG', 'x28brqtpu')
DATA = os.environ.get('DATA', '/kaggle/input/datasets/diegoaranguren/mscapital-matrices')
XTRA = os.environ.get('XTRA', '/kaggle/input/datasets/diegoaranguren/mscapital-x21')
XTRA2 = os.environ.get('XTRA2', '/kaggle/input/datasets/diegoaranguren/mscapital-x22')

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True
device = xm.xla_device() if XLA else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, 'seeds', SEEDS, 'XLA:', XLA, 'BS:', BS, flush=True)

# ---------- train features ----------
X_all = np.load(f'{DATA}/full_train.npy')
X1 = np.load(f'{XTRA}/X21_train.npy').astype(np.float32)
X2 = np.load(f'{XTRA2}/X22_train.npy').astype(np.float32)
X_all = np.concatenate([X_all, np.nan_to_num(X1), np.nan_to_num(X2)], axis=1)
DROP = [301, 267, 268, 299, 257]
X_all = np.delete(X_all, DROP, axis=1)
del X1, X2
y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32)
month = np.load(f'{DATA}/full_month.npy')

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
print('train 455f ->', X_all.shape, flush=True)

# ---------- test features ----------
Xte = np.load(f'{DATA}/full_test.npy')
names = np.load(f'{DATA}/full_names.npy', allow_pickle=True).tolist()
kin = {k: i for i, k in enumerate(names)}
nodata = (Xte[:, kin['X2:tx_n']] == 0) & (Xte[:, kin['X2:mk_nbars']] == 0)
X1e = np.load(f'{XTRA}/X21_test.npy').astype(np.float32)
X2e = np.load(f'{XTRA2}/X22_test.npy').astype(np.float32)
Xte = np.concatenate([Xte, np.nan_to_num(X1e), np.nan_to_num(X2e)], axis=1)
Xte = np.delete(Xte, DROP, axis=1)
del X1e, X2e
e726 = pd.read_csv(f'{d0726}/test.csv').sort_values('sample_id').reset_index(drop=True)
XE726 = np.nan_to_num(e726[fc].values.astype(np.float32))
assert XE726.shape[0] == Xte.shape[0]
Xte = np.concatenate([Xte, XE726], axis=1); del XE726, e726
print('test 455f ->', Xte.shape, flush=True)

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
        self.code_heads = nn.ModuleList([NTPLinear(n_ens, 128, 3) for _ in range(3)])
        mask = torch.ones(n_ens, total_dim, dtype=torch.bool)
        for i in range(n_ens):
            mask[i, i::n_ens // 2] = False
        self.register_buffer('feature_mask', mask)
    def forward(self, x_num, return_codes=False):
        x = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x = self.num_embed(x)
        x = x * self.feature_mask.unsqueeze(0).float()
        f = self.shared(x)
        reg = self.reg_head(f).squeeze(-1)   # (batch, n_ens)
        if return_codes:
            return reg, [h(f) for h in self.code_heads]
        return reg

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

from sklearn.cluster import KMeans
class RQKMeansEncoder:
    def __init__(self, n_layers=3, codebook_size=3):
        self.n_layers = n_layers; self.codebook_size = codebook_size; self.codebooks = []
    def fit(self, y):
        r = y.copy().reshape(-1, 1)
        for _ in range(self.n_layers):
            km = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10).fit(r)
            self.codebooks.append(km)
            r = r - km.cluster_centers_[km.predict(r)].reshape(-1, 1)
        return self
    def encode(self, y):
        r = y.copy().reshape(-1, 1); out = []
        for km in self.codebooks:
            c = km.predict(r); out.append(c)
            r = r - km.cluster_centers_[c].reshape(-1, 1)
        return np.stack(out, axis=1)

def rq_ce(code_logits, y_codes):
    import torch.nn.functional as F
    loss = 0.0
    for li, logits in enumerate(code_logits):
        labels = y_codes[:, li].unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
        loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)
    return loss / len(code_logits)

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

test_preds = np.zeros(len(Xte), dtype=np.float64)
oof = np.full(len(X_all), np.nan, dtype=np.float32)
oof_models = []   # per-model OOF vectors (NaN outside each model's val fold)
test_models = []  # per-model test predictions
fold_report = []
n_models = 0
t_all = time.time()

FOLD_IDX = set(int(x) for x in os.environ.get('FOLD_IDX', '4').split(','))
for seed in SEEDS:
    for fi, (va_mask, tr_mask) in enumerate(FOLDS):
        if fi not in FOLD_IDX: continue
        set_seed(seed * 100 + fi)
        med, fac = fit_scale(X_all[tr_mask])
        Xtr = apply_scale(X_all[tr_mask], med, fac)
        Xva = apply_scale(X_all[va_mask], med, fac)
        ytr = y_all[tr_mask]; yva = y_all[va_mask]
        rq = RQKMeansEncoder(n_layers=3, codebook_size=3).fit(np.round(ytr, 4))
        yc_t = torch.tensor(rq.encode(np.round(ytr, 4)), dtype=torch.long)
        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
        Xva_t = torch.tensor(Xva)
        if device.type in ('cuda','xla'):
            Xtr_t = Xtr_t.to(device); ytr_t = ytr_t.to(device); Xva_t = Xva_t.to(device)
            yc_t = yc_t.to(device)
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
                reg_b, logits_b = model(Xtr_t[idx], return_codes=True)
                loss = loss_fn(reg_b, by) + 0.1 * rq_ce(logits_b, yc_t[idx])
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
        print(f'seed {seed} fold {fi+1}: best val cos {best_cos:.6f} ({time.time()-t0:.0f}s)', flush=True)
        fold_report.append({'seed': seed, 'fold': fi+1, 'val_cos': float(best_cos), 's': round(time.time()-t0)})
        # predict with best EMA state
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xva_t), 4096):
                vp.append(model(Xva_t[i:i+4096]).mean(dim=1).cpu())
            _vp = torch.cat(vp).numpy()
            oof[va_mask] = _vp
            _om = np.full(len(X_all), np.nan, dtype=np.float32); _om[va_mask] = _vp
            oof_models.append(_om)
            Xte_s = apply_scale(Xte, med, fac)
            Xte_t = torch.tensor(Xte_s)
            if device.type in ('cuda','xla'): Xte_t = Xte_t.to(device)
            tp = []
            for i in range(0, len(Xte_t), 4096):
                tp.append(model(Xte_t[i:i+4096]).mean(dim=1).cpu())
            _tp = torch.cat(tp).numpy()
            test_preds += _tp
            test_models.append(_tp.astype(np.float32))
            n_models += 1
        # incremental save: partial results survive a session timeout/cancel
        np.save(f'/kaggle/working/oof_models_{TAG}_partial.npy', np.stack(oof_models))
        np.save(f'/kaggle/working/test_models_{TAG}_partial.npy', np.stack(test_models))
        _c = oof.copy()
        _seen = ~np.isnan(_c)
        _m = (month >= 61) & (month <= 70) & _seen
        json.dump({'n_models_done': n_models, 'folds': fold_report,
                   'oof_cos_61_70_partial': cos_np(_c[_m], y_all[_m]) if _m.any() else None},
                  open(f'/kaggle/working/metrics_{TAG}_partial.json','w'), indent=1)
        del model, opt, ema, Xtr_t, ytr_t, Xva_t, Xte_t, Xtr, Xva
        if device.type == 'cuda': torch.cuda.empty_cache()
        if XLA: xm.mark_step()

test_preds /= n_models
print('models averaged:', n_models, 'total time', round(time.time()-t_all), 's', flush=True)
np.save(f'/kaggle/working/test_pred_{TAG}.npy', test_preds.astype(np.float32))
np.save(f'/kaggle/working/oof_{TAG}.npy', oof)
np.save(f'/kaggle/working/oof_models_{TAG}.npy', np.stack(oof_models))
np.save(f'/kaggle/working/test_models_{TAG}.npy', np.stack(test_models))

# clipping bounds from fold-5 last model? use OOF-based bounds: predictions on seen train months (fold vals cover 40-70)
seen = ~np.isnan(oof)
lo, hi = np.quantile(oof[seen], 0.001), np.quantile(oof[seen], 0.999)
pte = test_preds.astype(np.float32).copy()
pte[nodata] = 0.0
pte = np.clip(pte, lo, hi)
cand = sorted(glob.glob('/kaggle/input/competitions/*/submission.csv'))
if cand:
    sub = pd.read_csv(cand[0])
    tcol = [c for c in sub.columns if c != 'sample_id'][0]
    sub[tcol] = pte
    sub.to_csv('/kaggle/working/submission.csv', index=False)
else:
    print('no competition submission.csv attached; skipping (validation run)', flush=True)

# metrics: OOF cos on months 61-70 (comparable to our champion val protocol)
m6170 = (month >= 61) & (month <= 70) & seen
late66 = (month >= 66) & (month <= 70) & seen
m4064 = (month >= 40) & (month <= 64) & seen
res = {'n_models': n_models, 'folds': fold_report,
       'oof_cos_61_70': cos_np(oof[m6170], y_all[m6170]),
       'oof_cos_66_70': cos_np(oof[late66], y_all[late66]),
       'oof_cos_40_64': cos_np(oof[m4064], y_all[m4064]),
       'elapsed_s': round(time.time()-t_all)}
json.dump(res, open(f'/kaggle/working/metrics_{TAG}.json','w'), indent=1)
print('OOF 61-70:', res['oof_cos_61_70'], '66-70:', res['oof_cos_66_70'], flush=True)
print('SUBMISSION_WRITTEN', pte.mean(), pte.std(), flush=True)
print('DONE', flush=True)
