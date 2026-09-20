# RealMLP champion recipe (455f) ported to JAX for Kaggle TPU (v3-8). tr<=60 / val 61-70.
# Replicates realmlp_val_x26.py minus X26: PBLD embedding, feature mask, NTP linears,
# grouped AdamW (hand-rolled), flat-anneal LR, label noise, EMA 0.998, cos+MSE loss.
# Validation target: val cos ~= 0.1671 (GPU champion). Saves val_pred + metrics to /kaggle/working.
import os, json, time, math
import numpy as np
import jax
import jax.numpy as jnp

SEED = 2026
N_ENS = 16
EPOCHS = 9
LR = 1e-3
BS = 256
TAG = os.environ.get('TAG', 'jax455')
TR_MAX, VA_LO, VA_HI = 60, 61, 70

def _resolve(fname):
    from pathlib import Path
    import subprocess
    r = subprocess.run(['find', '/kaggle/input', '-name', fname, '-maxdepth', '6'],
                       capture_output=True, text=True, timeout=90)
    ls = [l for l in r.stdout.strip().split('\n') if l.strip()]
    if ls: return str(Path(ls[0]).parent)
    raise FileNotFoundError(fname)

print('jax', jax.__version__, 'devices', jax.devices(), flush=True)
DATA = _resolve('full_train.npy'); XTRA = _resolve('X21_train.npy'); XTRA2 = _resolve('X22_train.npy')

X_all = np.load(f'{DATA}/full_train.npy')
Xt = np.load(f'{XTRA}/X21_train.npy').astype(np.float32)
X_all = np.concatenate([X_all, np.nan_to_num(Xt)], axis=1)
Xt2 = np.load(f'{XTRA2}/X22_train.npy').astype(np.float32)
X_all = np.concatenate([X_all, np.nan_to_num(Xt2)], axis=1)
DROP = [301, 267, 268, 299, 257]
X_all = np.delete(X_all, DROP, axis=1)
import pandas as pd
from pathlib import Path
d0726 = None
for cand in ['/kaggle/input/rfmf-0726data', '/kaggle/input/kernels/yunsuxiaozi/rfmf-0726data']:
    if Path(cand).exists() and (Path(cand)/'train.csv').exists():
        d0726 = cand; break
if d0726 is None:
    import subprocess
    r = subprocess.run(['find','/kaggle/input','-name','train.csv','-maxdepth','5'], capture_output=True, text=True, timeout=60)
    for l in r.stdout.strip().split('\n'):
        if l.strip(): d0726 = str(Path(l).parent); break
df = pd.read_csv(f'{d0726}/train.csv')
if 'sample_id' in df.columns:
    df = df.sort_values('sample_id').reset_index(drop=True)
dropc = ['sample_id','target','label','month','id']
num_cols = [c for c in df.columns if c not in dropc and df[c].dtype != object]
F = df[num_cols].to_numpy(np.float32)
F = np.nan_to_num(F, nan=0.0)
X_all = np.concatenate([X_all, F], axis=1)
del F, df
print('features:', X_all.shape, flush=True)

y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32)
month = np.load(f'{DATA}/full_month.npy')
tr = month <= TR_MAX; va = (month >= VA_LO) & (month <= VA_HI)
Xtr = X_all[tr]; ytr = y_all[tr]
Xva = X_all[va]; yva = y_all[va]
mva = month[va]
del X_all
print('tr', Xtr.shape, 'va', Xva.shape, flush=True)

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
n_feat = Xtr.shape[1]
emb_total = n_feat * 3

Xtr_j = jnp.array(Xtr); ytr_j = jnp.array(ytr)
Xva_j = jnp.array(Xva)
print('data on device', flush=True)

# ---------- init (match torch shapes) ----------
key = jax.random.PRNGKey(SEED)
k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
params = {
    'emb_w1': jax.random.normal(k1, (N_ENS, n_feat, 24)) * 1.0,
    'emb_b1': jax.random.uniform(k2, (N_ENS, n_feat, 24), minval=-np.pi, maxval=np.pi),
    'emb_w2': jax.random.normal(k3, (N_ENS, n_feat, 24, 2)) * (1.0 / np.sqrt(24)),
    'emb_b2': jax.random.normal(k4, (N_ENS, n_feat, 2)),
    'ln_w': jnp.ones((emb_total,)), 'ln_b': jnp.zeros((emb_total,)),
    'scale': jnp.ones((N_ENS, emb_total)),
    'l1_w': jax.random.normal(k5, (N_ENS, emb_total, 512)) * 0.02,
    'l1_b': jax.random.normal(k5, (N_ENS, 512)) * 0.02,
    'l2_w': jax.random.normal(k6, (N_ENS, 512, 512)) * 0.02,
    'l2_b': jax.random.normal(k6, (N_ENS, 512)) * 0.02,
    'l3_w': jax.random.normal(k7, (N_ENS, 512, 128)) * 0.02,
    'l3_b': jax.random.normal(k7, (N_ENS, 128)) * 0.02,
    'hd_w': jax.random.normal(k1, (N_ENS, 128, 1)) * 0.02,
    'hd_b': jax.random.normal(k2, (N_ENS, 1)) * 0.02,
}
mask = np.ones((N_ENS, emb_total), dtype=np.float32)
for i in range(N_ENS):
    mask[i, i::N_ENS // 2] = 0.0
mask_j = jnp.array(mask)

def gelu(x):
    return jax.nn.gelu(x, approximate=True)

def forward(params, x, train, rng):
    # x: (b, n_feat)
    b = x.shape[0]
    xb = jnp.broadcast_to(x[:, None, :], (b, N_ENS, n_feat))
    periodic = jnp.cos(2 * jnp.pi * (xb[..., None] * params['emb_w1'][None] + params['emb_b1'][None]))
    t = jnp.einsum('bnfh,nfhd->bnfd', periodic, params['emb_w2'])
    t = gelu(t + params['emb_b2'][None])
    h = jnp.concatenate([xb[..., None], t], axis=-1).reshape(b, N_ENS, emb_total)
    h = h * mask_j[None]
    mu = h.mean(-1, keepdims=True); sd = h.std(-1, keepdims=True)
    h = (h - mu) / (sd + 1e-5) * params['ln_w'][None, None] + params['ln_b'][None, None]
    h = h * params['scale'][None]
    def ntp(x, w, b_):
        return jnp.einsum('bni,nio->bno', x, w) / np.sqrt(x.shape[-1]) + b_[None]
    h = gelu(ntp(h, params['l1_w'], params['l1_b']))
    if train: h = jax.random.dropout(rng, 1 - 0.01, h)
    h = gelu(ntp(h, params['l2_w'], params['l2_b']))
    if train: h = jax.random.dropout(rng, 1 - 0.01, h)
    h = gelu(ntp(h, params['l3_w'], params['l3_b']))
    if train: h = jax.random.dropout(rng, 1 - 0.01, h)
    out = jnp.einsum('bni,nio->bno', h, params['hd_w']) / np.sqrt(128) + params['hd_b'][None]
    return out.squeeze(-1)   # (b, n_ens)

def loss_fn(params, x, y, rng):
    pred = forward(params, x, True, rng)
    ypf = pred.reshape(-1)
    ytf = jnp.broadcast_to(y[:, None], pred.shape).reshape(-1)
    w = jnp.where(jnp.abs(ytf) > 0.001, 0.5, 1.0)
    mse = (w * (ypf - ytf) ** 2).mean()
    pc = ypf - ypf.mean(); tc = ytf - ytf.mean()
    cos = (pc * tc).sum() / (jnp.linalg.norm(pc) + 1e-8) / (jnp.linalg.norm(tc) + 1e-8)
    return mse + (1 - cos)

# ---------- hand-rolled AdamW with param groups ----------
GROUP = {}
for k in params:
    if k == 'scale': GROUP[k] = 'scale'
    elif k.startswith('emb_'): GROUP[k] = 'pbld'
    elif k.endswith('_b') or k in ('ln_b',): GROUP[k] = 'bias'
    else: GROUP[k] = 'rest'
GR_LR = {'scale': LR * 20.0, 'pbld': LR * 0.093, 'rest': LR, 'bias': LR * 0.1}
GR_WD = {'scale': 1e-3, 'pbld': 1e-2, 'rest': 1e-2, 'bias': 5e-3}
B1, B2, EPS = 0.9, 0.98, 1e-8

def make_opt(params):
    return {'m': {k: jnp.zeros_like(v) for k, v in params.items()},
            'v': {k: jnp.zeros_like(v) for k, v in params.items()}, 't': 0}

def flat_anneal(v, progress, flat_ratio=0.5):
    return v if progress < flat_ratio else v * (1 - (progress - flat_ratio) / (1 - flat_ratio))

@jax.jit
def train_step(params, opt, x, y, rng, progress):
    def l(p):
        rng2 = jax.random.fold_in(rng, 1)
        ynoisy = y + jax.random.normal(rng, y.shape) * (0.005 * (1 - progress))
        return loss_fn(p, x, ynoisy, rng2)
    loss, grads = jax.value_and_grad(l)(params)
    t = opt['t'] + 1
    new_params, new_m, new_v = {}, {}, {}
    bc1 = 1 - B1 ** t; bc2 = 1 - B2 ** t
    for k in params:
        g = grads[k]
        m = B1 * opt['m'][k] + (1 - B1) * g
        v = B2 * opt['v'][k] + (1 - B2) * g ** 2
        mh = m / bc1; vh = v / bc2
        grp = GROUP[k]
        lr = flat_anneal(GR_LR[grp], float(progress))
        p = params[k] - lr * (mh / (jnp.sqrt(vh) + EPS) + GR_WD[grp] * params[k])
        new_params[k] = p; new_m[k] = m; new_v[k] = v
    return new_params, {'m': new_m, 'v': new_v, 't': t}, loss

@jax.jit
def ema_update(ema, params, decay=0.998):
    return {k: ema[k] * decay + params[k] * (1 - decay) for k in params}

@jax.jit
def predict(params, x):
    return forward(params, x, False, jax.random.PRNGKey(0)).mean(axis=1)

def cos_np(p, t):
    p = p - p.mean(); t = t - t.mean()
    return float((p * t).sum() / (np.linalg.norm(p) + 1e-8) / (np.linalg.norm(t) + 1e-8))

opt = make_opt(params)
ema = {k: v.copy() if hasattr(v, 'copy') else jnp.array(v) for k, v in params.items()}
ema = {k: jnp.array(v) for k, v in params.items()}
n = Xtr_j.shape[0]
steps_per_epoch = (n + BS - 1) // BS
total_steps = steps_per_epoch * EPOCHS
best_cos = -1; best_pv = None; best_ema = None
t0 = time.time()
key = jax.random.PRNGKey(SEED + 1)
for ep in range(EPOCHS):
    key, pk = jax.random.split(key)
    perm = np.array(jax.random.permutation(pk, n))
    for i in range(0, n, BS):
        idx = jnp.array(perm[i:i + BS])
        step = ep * steps_per_epoch + i // BS
        progress = min(step / total_steps, 1.0)
        key, sk = jax.random.split(key)
        params, opt, loss = train_step(params, opt, Xtr_j[idx], ytr_j[idx], sk, progress)
        ema = ema_update(ema, params)
    pv = np.array(predict(ema, Xva_j))
    c = cos_np(pv, yva)
    print(f'ep {ep+1}/{EPOCHS} val_cos {c:.6f} elapsed {time.time()-t0:.0f}s', flush=True)
    if c > best_cos:
        best_cos = c; best_pv = pv.copy()
        best_ema = {k: np.array(v) for k, v in ema.items()}
np.save(f'/kaggle/working/val_pred_{TAG}.npy', best_pv)
json.dump({'best_val_cos': best_cos, 'n_ens': N_ENS, 'epochs': EPOCHS, 'seed': SEED,
           'device': str(jax.devices()[0])}, open(f'/kaggle/working/metrics_{TAG}.json', 'w'))
print('DONE best val_cos', best_cos, flush=True)
