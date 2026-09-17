"""
TPU TabM cosine v6 — GPU-MATCHING config, 3 seeds.
v5 proved TPU pipeline works (31min, HBM peak 913MB) but overfit at h1024/k64/bs16384.
v6 matches GPU TabM cos689 architecture EXACTLY:
  - k=64, hidden=512, n_blocks=3, dropout=0.15
  - NO input BatchNorm (GPU version doesn't have it)
  - lr=1e-3, wd=1e-5, patience=15, grad_clip=1.0, cosine schedule
  - batch=8192 (GPU uses 2048; TPU needs larger batches for efficiency, v4 proved 8192 works)
  - 3 seeds [2026, 42, 7]
Reference GPU result: raw pool=0.16379, raw last=0.18697 (85min T4x2).
Expected TPU time: ~15-20min/seed, ~50-60min total.
"""
import os, sys, time, json, gc, subprocess, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
print("=== ktpu tabm cos v6 — GPU-matching config, 3 seeds ===", flush=True)
t0 = time.time()
TIME_LIMIT = 11 * 3600  # 11h hard limit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

DEVICE = xm.xla_device() if TPU_AVAILABLE else torch.device("cpu")
print(f"[TPU] device={DEVICE}", flush=True)

def hbm_info():
    if not TPU_AVAILABLE: return ""
    try:
        info = xm.get_memory_info(DEVICE)
        used = float(info.get("bytes_used", 0)) / 1024**2
        limit = float(info.get("bytes_limit", 0)) / 1024**2
        peak = float(info.get("peak_bytes_used", 0)) / 1024**2
        return f"[HBM] used={used:.0f}MB limit={limit:.0f}MB peak={peak:.0f}MB"
    except Exception as e:
        return f"[HBM] error: {e}"
print(hbm_info(), flush=True)

SEEDS = [2026, 42, 7]
K_HEADS = 64
HIDDEN = 512
N_BLOCKS = 3
DROPOUT = 0.15
EPOCHS = 60
BATCH_SIZE = 8192
LR = 1e-3
WD = 1e-5
PATIENCE = 15
GRAD_CLIP = 1.0
INFER_BS = 8192

# ============================================================
# Load features
# ============================================================
print("[DATA] loading...", flush=True)
OUR_DIR = None
for cand in ["/kaggle/input/mscapital-lgb-features",
             "/kaggle/input/datasets/bestwater/mscapital-lgb-features"]:
    if Path(cand).exists():
        OUR_DIR = cand; break
assert OUR_DIR is not None

X_train_ours = np.load(f"{OUR_DIR}/X_train_features.npy").astype(np.float32)
X_test_ours = np.load(f"{OUR_DIR}/X_test_features.npy").astype(np.float32)
train_ids = np.load(f"{OUR_DIR}/train_ids.npy")
test_ids = np.load(f"{OUR_DIR}/test_ids.npy")
train_months = np.load(f"{OUR_DIR}/train_months.npy").astype(np.int16)
train_target = np.load(f"{OUR_DIR}/train_target.npy").astype(np.float64)
print(f"[DATA] our features: {X_train_ours.shape} ({time.time()-t0:.0f}s)", flush=True)

DATA_DIR = None
for cand in ["/kaggle/input/rfmf-0726data",
             "/kaggle/input/datasets/yunsuxiaozi/rfmf-0726data",
             "/kaggle/input/yunsuxiaozi/rfmf-0726data",
             "/kaggle/input/notebooks/yunsuxiaozi/rfmf-0726data"]:
    p = Path(cand)
    if p.exists() and (p / "train.csv").exists():
        DATA_DIR = cand; break
if DATA_DIR is None:
    r = subprocess.run(["find", "/kaggle/input", "-name", "train.csv", "-maxdepth", "6"],
                       capture_output=True, text=True, timeout=30)
    for line in r.stdout.strip().split("\n"):
        if line and ("rfmf" in line.lower() or "0726" in line.lower()):
            DATA_DIR = str(Path(line).parent); break
assert DATA_DIR is not None
print(f"[DATA] 0726 dir: {DATA_DIR}", flush=True)

train_0726 = pd.read_csv(f"{DATA_DIR}/train.csv").sort_values("sample_id").reset_index(drop=True)
test_0726 = pd.read_csv(f"{DATA_DIR}/test.csv").sort_values("sample_id").reset_index(drop=True)
drop_cols = ["sample_id", "target", "label", "month", "id"]
feat_cols_0726 = [c for c in train_0726.columns if c not in drop_cols
                  and train_0726[c].dtype in ["float64","float32","int64","int32"]]
X_train_0726 = train_0726[feat_cols_0726].values.astype(np.float32)
X_test_0726 = test_0726[feat_cols_0726].values.astype(np.float32)
del train_0726, test_0726; gc.collect()
print(f"[DATA] 0726 features: {len(feat_cols_0726)}", flush=True)

print("[DATA] building XS features...", flush=True)
t_xs = time.time()
try:
    import lightgbm as lgb
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)
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

pct_train = np.zeros((X_train_ours.shape[0], 30), dtype=np.float32)
pct_test = np.zeros((X_test_ours.shape[0], 30), dtype=np.float32)
for i, f in enumerate(top30):
    for m in np.unique(train_months):
        mask = train_months == m
        vals = X_train_ours[mask, f]
        pct_train[mask, i] = (rankdata(vals) / mask.sum() - 0.5).astype(np.float32)
    pct_test[:, i] = (rankdata(X_test_ours[:, f]) / X_test_ours.shape[0] - 0.5).astype(np.float32)

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

def cross_rank_batch(data, feat_idx, batch_size=50000):
    n = data.shape[0]; nf = len(feat_idx)
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

X_train_all = np.concatenate([X_train_ours, X_train_0726, pct_train, z_train, cr_train], axis=1).astype(np.float32)
X_test_all = np.concatenate([X_test_ours, X_test_0726, pct_test, z_test, cr_test], axis=1).astype(np.float32)
del X_train_ours, X_test_ours, X_train_0726, X_test_0726
del pct_train, pct_test, z_train, z_test, cr_train, cr_test
gc.collect()
X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)
X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)
print(f"[DATA] combined: {X_train_all.shape} ({time.time()-t_xs:.0f}s)", flush=True)

y_raw = train_target.astype(np.float32)
FOLDS = [
    (train_months >= 40) & (train_months <= 44),
    (train_months >= 50) & (train_months <= 54),
    (train_months >= 55) & (train_months <= 59),
    (train_months >= 60) & (train_months <= 64),
    (train_months >= 65) & (train_months <= 70),
]
TRAIN_MASKS = [train_months<=37, train_months<=47, train_months<=52, train_months<=57, train_months<=62]
VALID = FOLDS[0]|FOLDS[1]|FOLDS[2]|FOLDS[3]|FOLDS[4]

mean = X_train_all[TRAIN_MASKS[-1]].mean(axis=0)
std = X_train_all[TRAIN_MASKS[-1]].std(axis=0) + 1e-8
X_train_all = (X_train_all - mean) / std
X_test_all = (X_test_all - mean) / std
n_feat = X_train_all.shape[1]
print(f"[DATA] n_feat={n_feat}, batch={BATCH_SIZE}, model=h{HIDDEN}_b{N_BLOCKS}_k{K_HEADS} (GPU-matching)", flush=True)

# ============================================================
# TabM model — EXACTLY matching GPU architecture (no input BN)
# ============================================================
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
    p = pred - pred.mean(); t = target - target.mean()
    return 1.0 - (p * t).sum() / (torch.norm(p) * torch.norm(t) + 1e-8)

# Pre-compute padded validation/test arrays for fixed-shape inference
n_te = len(X_test_all)
pad_te = ((n_te + INFER_BS - 1) // INFER_BS) * INFER_BS
Xte_pad = np.zeros((pad_te, n_feat), dtype=np.float32)
Xte_pad[:n_te] = X_test_all

# ============================================================
# Training — 3 seeds
# ============================================================
oof_all = np.zeros(len(train_ids), dtype=np.float32)
test_all = np.zeros(len(test_ids), dtype=np.float32)
all_results = {"model": "TabM_cosine_raw_689_tpu_v6", "n_features": n_feat,
               "config": {"k": K_HEADS, "hidden": HIDDEN, "n_blocks": N_BLOCKS,
                          "dropout": DROPOUT, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                          "lr": LR, "wd": WD, "patience": PATIENCE, "grad_clip": GRAD_CLIP,
                          "seeds": SEEDS, "infer_bs": INFER_BS},
               "per_seed": []}

for seed in SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    oof_seed = np.zeros(len(train_ids), dtype=np.float32)
    test_seed = np.zeros(len(test_ids), dtype=np.float32)
    fold_scores = []
    fold_epochs = []
    seed_t0 = time.time()

    for fi, (tr_mask, va_mask) in enumerate(zip(TRAIN_MASKS, FOLDS)):
        t_fold = time.time()
        if time.time() - t0 > TIME_LIMIT - 1800:
            print(f"[TIME] 30min reserve reached, stopping after F{fi}", flush=True)
            break

        Xtr_t = torch.from_numpy(X_train_all[tr_mask])
        ytr_t = torch.from_numpy(y_raw[tr_mask])
        Xva_np = X_train_all[va_mask]
        yva_raw = y_raw[va_mask]
        n_va = len(Xva_np)
        pad_n = ((n_va + INFER_BS - 1) // INFER_BS) * INFER_BS
        Xva_pad = np.zeros((pad_n, n_feat), dtype=np.float32)
        Xva_pad[:n_va] = Xva_np

        train_ds = TensorDataset(Xtr_t, ytr_t)
        train_loader_raw = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, drop_last=True)
        if TPU_AVAILABLE:
            train_loader = pl.MpDeviceLoader(train_loader_raw, DEVICE)
        else:
            train_loader = train_loader_raw

        model = TabM(n_feat, K_HEADS, HIDDEN, N_BLOCKS, DROPOUT).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        best_cos = -1; best_state = None; patience_cnt = 0; best_ep = 0
        n_batches = len(train_loader_raw)

        for ep in range(EPOCHS):
            t_ep = time.time()
            model.train()
            ep_loss = 0.0
            for bi, (xb, yb) in enumerate(train_loader):
                pred = model(xb)
                loss = cosine_loss(pred, yb)
                opt.zero_grad(); loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if TPU_AVAILABLE:
                    xm.optimizer_step(opt, barrier=False)
                    xm.mark_step()
                else:
                    opt.step()
                ep_loss += loss.item()
            sched.step()

            # Validation — fixed batch size, padded
            model.eval()
            vp = []
            with torch.no_grad():
                for s in range(0, pad_n, INFER_BS):
                    xb = torch.from_numpy(Xva_pad[s:s+INFER_BS]).to(DEVICE)
                    vp.append(model(xb).cpu().numpy())
                    if TPU_AVAILABLE: xm.mark_step()
            vp = np.concatenate(vp)[:n_va]
            cs = cos_np(yva_raw, vp)
            ep_time = time.time() - t_ep

            improved = cs > best_cos
            if improved:
                best_cos = cs; best_ep = ep + 1; patience_cnt = 0
                best_state = {k.replace("module.", ""): v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            print(f"  F{fi+1} E{ep+1:02d} val={cs:.5f} best={best_cos:.5f} "
                  f"loss={ep_loss/n_batches:.4f} {'*' if improved else ' '} "
                  f"pat={patience_cnt}/{PATIENCE} {ep_time:.0f}s {hbm_info()}", flush=True)

            if patience_cnt >= PATIENCE:
                print(f"  F{fi+1} early stop at epoch {ep+1}, best epoch {best_ep}", flush=True)
                break

        # Load best and predict
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = []
            for s in range(0, pad_n, INFER_BS):
                xb = torch.from_numpy(Xva_pad[s:s+INFER_BS]).to(DEVICE)
                vp.append(model(xb).cpu().numpy())
                if TPU_AVAILABLE: xm.mark_step()
            oof_seed[va_mask] = np.concatenate(vp)[:n_va]

            tp = []
            for s in range(0, pad_te, INFER_BS):
                xb = torch.from_numpy(Xte_pad[s:s+INFER_BS]).to(DEVICE)
                tp.append(model(xb).cpu().numpy())
                if TPU_AVAILABLE: xm.mark_step()
            test_seed += np.concatenate(tp)[:n_te] / 5.0

        fold_scores.append(best_cos)
        fold_epochs.append(best_ep)
        print(f"  F{fi+1} BEST: {best_cos:.5f} at epoch {best_ep} ({time.time()-t_fold:.0f}s)", flush=True)
        del model, Xtr_t, ytr_t, train_ds, train_loader_raw, best_state, Xva_pad
        if TPU_AVAILABLE:
            del train_loader; xm.mark_step()
        gc.collect()

    # Seed summary
    n_done = len(fold_scores)
    if n_done == 5:
        seed_pool = cos_np(y_raw[VALID], oof_seed[VALID])
        seed_last = cos_np(y_raw[FOLDS[4]], oof_seed[FOLDS[4]])
    else:
        done_mask = FOLDS[0]
        for j in range(1, n_done): done_mask = done_mask | FOLDS[j]
        seed_pool = cos_np(y_raw[done_mask], oof_seed[done_mask])
        seed_last = fold_scores[-1] if fold_scores else 0

    seed_elapsed = time.time() - seed_t0
    print(f"  Seed {seed}: pool={seed_pool:.5f} last={seed_last:.5f} "
          f"folds={[f'{s:.5f}' for s in fold_scores]} epochs={fold_epochs} "
          f"({seed_elapsed:.0f}s)", flush=True)

    all_results["per_seed"].append({
        "seed": seed, "pool_cos": float(seed_pool), "last_cos": float(seed_last),
        "fold_scores": [float(s) for s in fold_scores],
        "fold_best_epochs": fold_epochs, "time_s": float(seed_elapsed)
    })
    oof_all += oof_seed / len(SEEDS)
    test_all += test_seed / len(SEEDS)

# ============================================================
# Final results
# ============================================================
final_pool = cos_np(y_raw[VALID], oof_all[VALID])
final_last = cos_np(y_raw[FOLDS[4]], oof_all[FOLDS[4]])
total_time = time.time() - t0

print(f"\n{'='*60}", flush=True)
print(f"=== TPU v6 FINAL RESULTS (GPU-matching, {len(SEEDS)} seeds) ===", flush=True)
print(f"  RAW pool: {final_pool:.5f}", flush=True)
print(f"  RAW last: {final_last:.5f}", flush=True)
for ps in all_results["per_seed"]:
    print(f"  Seed {ps['seed']}: pool={ps['pool_cos']:.5f} last={ps['last_cos']:.5f}", flush=True)
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.2f}h)", flush=True)
print(f"  {hbm_info()}", flush=True)
print(f"  Reference GPU TabM cos689 3-seed: pool=0.16379 last=0.18697", flush=True)
print(f"  Delta vs GPU: pool={final_pool-0.16379:+.5f} last={final_last-0.18697:+.5f}", flush=True)
print(f"{'='*60}", flush=True)

all_results["final_pool_cos"] = float(final_pool)
all_results["final_last_cos"] = float(final_last)
all_results["total_time_s"] = float(total_time)

np.save("oof_tpu_v6.npy", np.column_stack([train_ids, oof_all]))
np.save("test_tpu_v6.npy", np.column_stack([test_ids, test_all]))
with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)
pd.DataFrame({"sample_id": test_ids, "prediction": test_all}).to_csv("submission.csv", index=False)
print("DONE", flush=True)
