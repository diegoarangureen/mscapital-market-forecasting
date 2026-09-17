# X23: tabular aggregates of the seq3 order-flow stream (120s x 12ch grid, last step = most recent).
# 12 stats per channel (mean,std,min,max,q10,q90,mean_last30,mean_first30,diff_30,std_last30,ac1,ac5)
# + 4 cross-channel per-sample correlations = 148 features. CPU kernel, block-wise from memmap.
import os, time, json
import numpy as np

SEQTR = os.environ.get('SEQTR', '/kaggle/input/datasets/diegoaranguren/mscapital-seq3-train')
SEQTE = os.environ.get('SEQTE', '/kaggle/input/datasets/diegoaranguren/mscapital-seq3-test')
NS = {'train': 1257637, 'test': 647896}
STEPS, CH = 120, 12
BS = 50000

def stats_block(g):  # g: (n, 120, 12) float32
    n = len(g)
    mu = g.mean(axis=1); sd = g.std(axis=1)
    mn = g.min(axis=1); mx = g.max(axis=1)
    q10 = np.quantile(g, 0.10, axis=1); q90 = np.quantile(g, 0.90, axis=1)
    last = g[:, -30:, :]; first = g[:, :30, :]
    ml = last.mean(axis=1); mf = first.mean(axis=1)
    dl = ml - mf; sl = last.std(axis=1)
    c = g - mu[:, None, :]
    denom = (c ** 2).sum(axis=1) + 1e-12
    ac1 = (c[:, 1:, :] * c[:, :-1, :]).sum(axis=1) / denom
    ac5 = (c[:, 5:, :] * c[:, :-5, :]).sum(axis=1) / denom
    out = np.concatenate([mu, sd, mn, mx, q10, q90, ml, mf, dl, sl, ac1, ac5], axis=1)  # (n, 144)
    # cross-channel per-sample correlations
    pairs = [(0, 5), (0, 2), (5, 2), (1, 6)]
    xc = np.empty((n, len(pairs)), np.float32)
    for k, (i, j) in enumerate(pairs):
        a = g[:, :, i] - mu[:, None, i]; b = g[:, :, j] - mu[:, None, j]
        xc[:, k] = (a * b).sum(axis=1) / (np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-12)
    return np.concatenate([out, xc], axis=1)  # (n, 148)

for split, path in [('train', f'{SEQTR}/seq3_train.f16'), ('test', f'{SEQTE}/seq3_test.f16')]:
    t0 = time.time()
    mm = np.memmap(path, dtype=np.float16, mode='r', shape=(NS[split], STEPS, CH))
    out = np.empty((NS[split], 148), np.float32)
    for lo in range(0, NS[split], BS):
        hi = min(lo + BS, NS[split])
        g = np.asarray(mm[lo:hi], dtype=np.float32)
        out[lo:hi] = stats_block(g)
        del g
        if lo % 200000 == 0: print(split, lo, f'{time.time()-t0:.0f}s', flush=True)
    np.save(f'/kaggle/working/X23_{split}.npy', out)
    print(split, 'done', out.shape, f'{time.time()-t0:.0f}s', flush=True)
print('DONE', flush=True)
