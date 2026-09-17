# %% [code]
"""
GPU TabM with COSINE LOSS on RAW target, 689 features (462+152+75 XS), 3 seeds, 5-fold S2.
Combines two proven improvements:
  1. Cosine loss on raw target (E-C75: +0.020 raw pool vs RN+MSE on 614)
  2. XS cross-sectional features (E-C76: +0.006 raw pool vs 614 on RN+MSE)
Architecture identical to TabM614/689; only loss/target + feature set changed.
drop_last for DataParallel BatchNorm safety (v4 fix).
"""
import os, sys, time, json, gc, subprocess, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
print("=== GPU TabM COSINE LOSS on RAW target (689 feat, 3-seed, T4x2) ===", flush=True)
t0 = time.time()

SEEDS = [2026, 42, 7]
K_HEADS = 64
HIDDEN = 512
N_BLOCKS = 3
DROPOUT = 0.15
EPOCHS = 60
BATCH_SIZE = 2048
LR = 1e-3
WD = 1e-5
PATIENCE = 15
GRAD_CLIP = 1.0

# ============================================================
# Load our 462 features
# ============================================================
print("Loading our features...", flush=True)
OUR_DIR = None
for cand in ["/kaggle/input/mscapital-lgb-features",
             "/kaggle/input/datasets/bestwater/mscapital-lgb-features"]:
    if Path(cand).exists():
        OUR_DIR = cand; break
assert OUR_DIR is not None, "Our features not found"

X_train_ours = np.load(f"{OUR_DIR}/X_train_features.npy").astype(np.float32)
X_test_ours = np.load(f"{OUR_DIR}/X_test_features.npy").astype(np.float32)
train_ids = np.load(f"{OUR_DIR}/train_ids.npy")
test_ids = np.load(f"{OUR_DIR}/test_ids.npy")
train_months = np.load(f"{OUR_DIR}/train_months.npy").astype(np.int16)
train_target = np.load(f"{OUR_DIR}/train_target.npy").astype(np.float64)
print(f"Our features: train={X_train_ours.shape}, test={X_test_ours.shape}", flush=True)

# ============================================================
# Load 0726 features (152)
# ============================================================
print("Loading 0726 features...", flush=True)
DATA_DIR = None
for cand in ["/kaggle/input/rfmf-0726data",
             "/kaggle/input/datasets/yunsuxiaozi/rfmf-0726data",
             "/kaggle/input/yunsuxiaozi/rfmf-0726data"]:
    p = Path(cand)
    if p.exists() and (p / "train.csv").exists():
        DATA_DIR = cand; break
if DATA_DIR is None:
    r = subprocess.run(["find", "/kaggle/input", "-name", "train.csv", "-maxdepth", "5"],
                       capture_output=True, text=True, timeout=30)
    for line in r.stdout.strip().split("\n"):
        if "rfmf" in line.lower() or "0726" in line.lower():
            DATA_DIR = str(Path(line).parent); break
assert DATA_DIR is not None, "0726 train.csv not found"
print(f"0726 data dir: {DATA_DIR}", flush=True)

train_0726 = pd.read_csv(f"{DATA_DIR}/train.csv").sort_values("sample_id").reset_index(drop=True)
test_0726 = pd.read_csv(f"{DATA_DIR}/test.csv").sort_values("sample_id").reset_index(drop=True)
assert (train_0726["sample_id"].values == train_ids).all(), "Train ID mismatch"
assert (test_0726["sample_id"].values == test_ids).all(), "Test ID mismatch"

drop_cols = ["sample_id", "target", "label", "month", "id"]
feat_cols_0726 = [c for c in train_0726.columns if c not in drop_cols
                  and train_0726[c].dtype in ["float64","float32","int64","int32"]]
X_train_0726 = train_0726[feat_cols_0726].values.astype(np.float32)
X_test_0726 = test_0726[feat_cols_0726].values.astype(np.float32)
del train_0726, test_0726; gc.collect()
print(f"0726 features: {len(feat_cols_0726)}", flush=True)

# ============================================================
# Build 75 XS cross-sectional features from 462 base
# ============================================================
print("Building XS cross-sectional features...", flush=True)
t_xs = time.time()

try:
    import lightgbm as lgb
    np.random.seed(42)
    idx = np.random.choice(X_train_ours.shape[0], 200000, replace=False)
    tmp_params = dict(objective="regression", metric="rmse", learning_rate=0.1, num_leaves=63,
                      min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=5, verbose=-1, seed=42, n_jobs=-1)
    d_tmp = lgb.Dataset(X_train_ours[idx], train_target[idx])
    m_tmp = lgb.train(tmp_params, d_tmp, num_boost_round=100)
    top30 = np.argsort(m_tmp.feature_importance(importance_type="gain"))[::-1][:30]
    top15 = top30[:15]
    del d_tmp, m_tmp, idx; gc.collect()
    print(f"  Top-30 features selected via LGB ({time.time()-t_xs:.0f}s)", flush=True)
except Exception as e:
    print(f"  LGB top-feature selection failed ({e}), using first 30 features", flush=True)
    top30 = np.arange(30)
    top15 = np.arange(15)

# 1. Within-month percentile rank (top-30)
pct_train = np.zeros((X_train_ours.shape[0], 30), dtype=np.float32)
pct_test = np.zeros((X_test_ours.shape[0], 30), dtype=np.float32)
for i, f in enumerate(top30):
    for m in np.unique(train_months):
        mask = train_months == m
        vals = X_train_ours[mask, f]
        pct_train[mask, i] = (rankdata(vals) / mask.sum() - 0.5).astype(np.float32)
    pct_test[:, i] = (rankdata(X_test_ours[:, f]) / X_test_ours.shape[0] - 0.5).astype(np.float32)
print(f"  Percentile rank done ({time.time()-t_xs:.0f}s)", flush=True)

# 2. Within-month z-score (top-15)
z_train = np.zeros((X_train_ours.shape[0], 15), dtype=np.float32)
z_test = np.zeros((X_test_ours.shape[0], 15), dtype=np.float32)
for i, f in enumerate(top15):
    for m in np.unique(train_months):
        mask = train_months == m
        vals = X_train_ours[mask, f]
        mu, sd = vals.mean(), vals.std() + 1e-8
        z_train[mask, i] = ((vals - mu) / sd).astype(np.float32)
    mu, sd = X_test_ours[:, f].mean(), X_test_ours[:, f].std() + 1e-8
    z_test[:, i] = ((X_test_ours[:, f] - mu) / sd).astype(np.float32)
print(f"  Z-score done ({time.time()-t_xs:.0f}s)", flush=True)

# 3. Cross-feature rank (top-30)
def cross_rank_batch(data, feat_idx, batch_size=50000):
    n = data.shape[0]
    nf = len(feat_idx)
    out = np.zeros((n, nf), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s+batch_size, n)
        batch = data[s:e][:, feat_idx].astype(np.float32)
        order = np.argsort(batch, axis=1)
        ranks = np.empty_like(order, dtype=np.float32)
        cols = np.arange(nf) / max(nf-1, 1)
        for j in range(e-s):
            ranks[j, order[j]] = cols
        out[s:e] = ranks - 0.5
    return out

cr_train = cross_rank_batch(X_train_ours, top30)
cr_test = cross_rank_batch(X_test_ours, top30)
print(f"  Cross-rank done ({time.time()-t_xs:.0f}s)", flush=True)

# Combine: 462 + 152 + 30 + 15 + 30 = 689
X_train_all = np.concatenate([X_train_ours, X_train_0726, pct_train, z_train, cr_train], axis=1).astype(np.float32)
X_test_all = np.concatenate([X_test_ours, X_test_0726, pct_test, z_test, cr_test], axis=1).astype(np.float32)
del X_train_ours, X_test_ours, X_train_0726, X_test_0726
del pct_train, pct_test, z_train, z_test, cr_train, cr_test
gc.collect()
X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)
X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Combined: train={X_train_all.shape} test={X_test_all.shape} ({time.time()-t_xs:.0f}s)", flush=True)

# ============================================================
# Targets: RAW target for training (cosine loss), RN for comparison
# ============================================================
def rank_norm_per_month(v, m):
    r = np.zeros_like(v, dtype=np.float64)
    for mo in np.unique(m):
        mask = m == mo
        if mask.sum() >= 2:
            r[mask] = (rankdata(v[mask]) / mask.sum() - 0.5) * 2.0
    return r

y_raw = train_target.astype(np.float32)
y_rn = rank_norm_per_month(train_target, train_months)

FOLDS = [
    (train_months >= 40) & (train_months <= 44),
    (train_months >= 50) & (train_months <= 54),
    (train_months >= 55) & (train_months <= 59),
    (train_months >= 60) & (train_months <= 64),
    (train_months >= 65) & (train_months <= 70),
]
TRAIN_MASKS = [train_months<=37, train_months<=47, train_months<=52, train_months<=57, train_months<=62]
VALID = FOLDS[0]|FOLDS[1]|FOLDS[2]|FOLDS[3]|FOLDS[4]

# Standardize using training data
mean = X_train_all[TRAIN_MASKS[-1]].mean(axis=0)
std = X_train_all[TRAIN_MASKS[-1]].std(axis=0) + 1e-8
X_train_all = (X_train_all - mean) / std
X_test_all = (X_test_all - mean) / std

# ============================================================
# PyTorch
# ============================================================
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    try:
        _t = torch.zeros(1, device="cuda"); _ = _t + 1
        torch.cuda.synchronize(); del _t; torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
        print(f"Device: {DEVICE}, {n_gpu} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)
    except Exception as e:
        print(f"CUDA failed ({e}), falling back to CPU", flush=True)
        DEVICE = torch.device("cpu"); n_gpu = 0
else:
    n_gpu = 0
    print("Device: CPU", flush=True)

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
    """1 - cosine similarity. Differentiable. Scale-invariant."""
    p = pred - pred.mean()
    t = target - target.mean()
    return 1.0 - (p * t).sum() / (torch.norm(p) * torch.norm(t) + 1e-8)

n_feat = X_train_all.shape[1]
print(f"n_feat={n_feat}, batch_size={BATCH_SIZE} x {n_gpu} GPU(s), seeds={SEEDS}", flush=True)
print(f"Training target: RAW, Loss: COSINE, Early stop on: raw cosine", flush=True)

# ============================================================
# Training loop
# ============================================================
oof_all = np.zeros(len(train_ids), dtype=np.float32)
test_all = np.zeros(len(test_ids), dtype=np.float32)
results = {"model": "TabM_cosine_raw_689", "n_features": n_feat,
           "loss": "cosine", "target": "raw", "per_seed": []}

for seed in SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    oof_seed = np.zeros(len(train_ids), dtype=np.float32)
    test_seed = np.zeros(len(test_ids), dtype=np.float32)
    fold_raw = []; fold_rn = []
    seed_t0 = time.time()

    for fi, (tr_mask, va_mask) in enumerate(zip(TRAIN_MASKS, FOLDS)):
        t_fold = time.time()
        Xtr = torch.from_numpy(X_train_all[tr_mask]).to(DEVICE)
        ytr = torch.from_numpy(y_raw[tr_mask]).to(DEVICE)
        Xva = torch.from_numpy(X_train_all[va_mask]).to(DEVICE)
        yva_raw = y_raw[va_mask]
        yva_rn = y_rn[va_mask]

        model = TabM(n_feat, K_HEADS, HIDDEN, N_BLOCKS, DROPOUT).to(DEVICE)
        if n_gpu > 1:
            model = nn.DataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        best_cos = -1; best_state = None; patience_cnt = 0
        n_tr = len(Xtr)
        n_full = (n_tr // BATCH_SIZE) * BATCH_SIZE  # drop_last for DataParallel BN safety

        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(n_tr)
            for s in range(0, n_full, BATCH_SIZE):
                idx = perm[s:s+BATCH_SIZE]
                xb = Xtr[idx]; yb = ytr[idx]
                pred = model(xb).squeeze()
                loss = cosine_loss(pred, yb)
                opt.zero_grad(); loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vp = []
                for s in range(0, len(Xva), 8192):
                    vp.append(model(Xva[s:s+8192]).squeeze().cpu().numpy())
                vp = np.concatenate(vp)

            cs_raw = cos_np(yva_raw, vp)
            if cs_raw > best_cos:
                best_cos = cs_raw
                best_state = {k.replace("module.", ""): v.cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE: break

        if n_gpu > 1:
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = []
            for s in range(0, len(Xva), 8192):
                vp.append(model(Xva[s:s+8192]).squeeze().cpu().numpy())
            oof_seed[va_mask] = np.concatenate(vp)

            tp = []
            Xte = torch.from_numpy(X_test_all).to(DEVICE)
            for s in range(0, len(Xte), 8192):
                tp.append(model(Xte[s:s+8192]).squeeze().cpu().numpy())
            test_seed += np.concatenate(tp) / 5.0

        best_rn = cos_np(yva_rn, oof_seed[va_mask])
        fold_raw.append(best_cos); fold_rn.append(best_rn)
        elapsed = time.time() - t_fold
        print(f"  F{fi+1} BEST: raw={best_cos:.5f} rn={best_rn:.5f} time={elapsed:.0f}s", flush=True)
        del model, Xtr, ytr, Xva, best_state; torch.cuda.empty_cache(); gc.collect()

    seed_raw_pool = cos_np(y_raw[VALID], oof_seed[VALID])
    seed_raw_last = cos_np(y_raw[FOLDS[4]], oof_seed[FOLDS[4]])
    seed_rn_pool = cos_np(y_rn[VALID], oof_seed[VALID])
    seed_rn_last = cos_np(y_rn[FOLDS[4]], oof_seed[FOLDS[4]])
    seed_elapsed = time.time() - seed_t0
    print(f"  Seed {seed}: RAW pool={seed_raw_pool:.5f} last={seed_raw_last:.5f} | "
          f"RN pool={seed_rn_pool:.5f} last={seed_rn_last:.5f} | {seed_elapsed:.0f}s", flush=True)
    results["per_seed"].append({"seed": seed, "raw_pool": float(seed_raw_pool), "raw_last": float(seed_raw_last),
                                 "rn_pool": float(seed_rn_pool), "rn_last": float(seed_rn_last),
                                 "fold_raw": [float(s) for s in fold_raw],
                                 "fold_rn": [float(s) for s in fold_rn],
                                 "time_s": float(seed_elapsed)})
    oof_all += oof_seed / len(SEEDS)
    test_all += test_seed / len(SEEDS)

# ============================================================
# Final evaluation
# ============================================================
final_raw_pool = cos_np(y_raw[VALID], oof_all[VALID])
final_raw_last = cos_np(y_raw[FOLDS[4]], oof_all[FOLDS[4]])
final_rn_pool = cos_np(y_rn[VALID], oof_all[VALID])
final_rn_last = cos_np(y_rn[FOLDS[4]], oof_all[FOLDS[4]])
total_time = time.time() - t0

results["final_raw_pool_cos"] = float(final_raw_pool)
results["final_raw_last_cos"] = float(final_raw_last)
results["final_rn_pool_cos"] = float(final_rn_pool)
results["final_rn_last_cos"] = float(final_rn_last)
results["elapsed_s"] = float(total_time)
results["config"] = {"k": K_HEADS, "hidden": HIDDEN, "n_blocks": N_BLOCKS, "dropout": DROPOUT,
                     "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "n_features": n_feat,
                     "n_gpu": n_gpu, "seeds": SEEDS, "loss": "cosine", "target": "raw",
                     "grad_clip": GRAD_CLIP}

print(f"\n{'='*60}", flush=True)
print(f"=== FINAL RESULTS (TabM COSINE on RAW, 689 feat, {len(SEEDS)} seeds) ===", flush=True)
print(f"  RAW pred vs RAW target: pool={final_raw_pool:.5f} last={final_raw_last:.5f}  <-- LB-like", flush=True)
print(f"  RAW pred vs RN target:  pool={final_rn_pool:.5f} last={final_rn_last:.5f}", flush=True)
for ps in results["per_seed"]:
    print(f"  Seed {ps['seed']}: raw_pool={ps['raw_pool']:.5f} raw_last={ps['raw_last']:.5f}", flush=True)
print(f"  Total time: {total_time/3600:.2f}h", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Reference cos614 3-seed: RAW pool=0.15880 last=0.17838", flush=True)
print(f"  Reference TabM689 RN+MSE 3-seed: RAW pool=0.14457 last=0.15951", flush=True)
print(f"  Delta vs cos614: pool={final_raw_pool-0.15880:+.5f} last={final_raw_last-0.17838:+.5f}", flush=True)

np.save("oof_tabm_cos689.npy", np.column_stack([train_ids, oof_all]))
np.save("test_tabm_cos689.npy", np.column_stack([test_ids, test_all]))
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame({"sample_id": test_ids, "prediction": test_all}).to_csv("submission.csv", index=False)
print(f"Saved. Total: {total_time/3600:.2f}h", flush=True)
print("DONE")
