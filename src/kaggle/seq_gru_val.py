# GRU over 120x8 bar sequences (market streams, per-second). GPU kernel. Val protocol: tr<=60 -> va 61-70 (+late 66-70).
import os, math, json, time
import numpy as np
import torch
import torch.nn as nn

SEED = int(os.environ.get('SEED', '2026'))
EPOCHS = int(os.environ.get('EPOCHS', '16'))
LR = float(os.environ.get('LR', '2e-3'))
BS = int(os.environ.get('BS', '1024'))
HID = int(os.environ.get('HID', '96'))
TR_MAX = int(os.environ.get('TR_MAX', '60'))
VA_LO = int(os.environ.get('VA_LO', '61'))
TAG = os.environ.get('TAG', 'seq2')
SEQDATA = os.environ.get('SEQDATA', '/kaggle/input/datasets/diegoaranguren/mscapital-seq2')
DATA = os.environ.get('DATA', '/kaggle/input/datasets/diegoaranguren/mscapital-matrices')

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

NS_TR, NS_TE, STEPS, CH = 1257637, 647896, 120, 8
X_all = np.memmap(f'{SEQDATA}/seq2_train.f16', dtype=np.float16, mode='r', shape=(NS_TR, STEPS, CH))
y_all = np.load(f'{DATA}/full_y.npy').astype(np.float32) * 1000.0
month = np.load(f'{DATA}/full_month.npy')
tr = month <= TR_MAX; va = month >= VA_LO
late = month >= 66
tr_idx = np.where(tr)[0]; va_idx = np.where(va)[0]
yv = y_all[va]; late_mask = late[va]
print('tr', len(tr_idx), 'va', len(va_idx), flush=True)

# channel stats from train rows
rng = np.random.default_rng(0)
sub = np.asarray(X_all[rng.choice(tr_idx, 40000, replace=False)], dtype=np.float32)
mu = sub.reshape(-1, CH).mean(0); sd = sub.reshape(-1, CH).std(0) + 1e-6
del sub
mu_t = torch.tensor(mu).to(device); sd_t = torch.tensor(sd).to(device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(CH, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
        )
        self.gru = nn.GRU(32, HID, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.Linear(HID, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        h, _ = self.gru(x)
        return self.head(h[:, -1]).squeeze(-1)

def load_batch(idx):
    xb = np.asarray(X_all[np.sort(idx)], dtype=np.float32)
    xb = torch.from_numpy(xb).to(device)
    xb = (xb - mu_t) / sd_t
    return torch.clamp(xb, -8, 8)

def cos_np(p, t):
    p = p - p.mean(); t = t - t.mean()
    return float(p @ t / (np.linalg.norm(p) + 1e-30) / (np.linalg.norm(t) + 1e-30))

model = Net().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * ((len(tr_idx)+BS-1)//BS))
yt = torch.from_numpy(y_all).to(device)

def evaluate():
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(va_idx), 4096):
            preds.append(model(load_batch(va_idx[i:i+4096])).cpu())
    return torch.cat(preds).numpy()

best = -1; best_pv = None; t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(tr_idx))
    for i in range(0, len(tr_idx), BS):
        idx = tr_idx[perm[i:i+BS].numpy()]
        xb = load_batch(idx)
        pred = model(xb)
        loss = ((pred - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    pv = evaluate()
    c_full = cos_np(pv, yv); c_late = cos_np(pv[late_mask], yv[late_mask])
    print(f'ep {ep+1}/{EPOCHS} val_cos {c_full:.6f} late_cos {c_late:.6f} elapsed {time.time()-t0:.0f}s', flush=True)
    if c_full > best:
        best = c_full; best_pv = pv.copy()
        torch.save(model.state_dict(), f'/kaggle/working/seq_best_{TAG}.pt')
np.save(f'/kaggle/working/val_pred_{TAG}.npy', best_pv)
json.dump({'best_val_cos': best, 'hid': HID, 'epochs': EPOCHS, 'seed': SEED}, open(f'/kaggle/working/metrics_{TAG}.json','w'))
print('DONE best val_cos', best, flush=True)
