# Build standardized 181-feat MLP matrix (58 X2 + 59 fp60R logdiff + 60 fp60V log1p + 4 X6), sequential loads.
import numpy as np, gc, time
import pyarrow.feather as feather
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split('kB')[0])//1024
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
y = lab.target.values.astype(np.float64)*1000.0; month = lab.month.values
yv = lab.target.values[month>=61].copy()
del lab; gc.collect()
n_total = len(y); NG = 60
X2 = np.load('/tmp/work/X2_train.npy'); d2 = X2.shape[1]
D = d2 + (NG-1) + NG + 4
X = np.empty((n_total, D), np.float32)
X[:, :d2] = X2; del X2; gc.collect()
print('X2 in, rss', rss(), flush=True)
BB = 200000
fpR = np.load('/tmp/work/fp60R_train.npy')
for st in range(0, n_total, BB):
    en = min(st+BB, n_total)
    blk = np.maximum(fpR[st:en], 1e-9); np.log(blk, out=blk)
    X[st:en, d2:d2+NG-1] = np.diff(blk, axis=1); del blk
del fpR; gc.collect(); __import__("ctypes").CDLL("libc.so.6").malloc_trim(0)
print('fpR in, rss', rss(), flush=True)
fpV = np.load('/tmp/work/fp60V_train.npy')
for st in range(0, n_total, BB):
    en = min(st+BB, n_total)
    blk = fpV[st:en]; np.log1p(blk, out=blk)
    X[st:en, d2+NG-1:d2+2*NG-1] = blk; del blk
del fpV; gc.collect(); __import__("ctypes").CDLL("libc.so.6").malloc_trim(0)
print('fpV in, rss', rss(), flush=True)
X6 = np.load('/tmp/work/X6_train.npy')
X[:, -4:] = X6; del X6; gc.collect()
for st in range(0, n_total, 100000):
    blk = X[st:st+100000]
    m = np.isnan(blk); blk[m] = 0.0
    np.clip(blk, -1e6, 1e6, out=blk)
    del m
B = 100000
tr = month<=60
s1 = np.zeros(D); s2 = np.zeros(D); ntr = 0
for st in range(0, len(X), B):
    blk = X[st:st+B]; m = tr[st:st+B]; bm = blk[m]
    s1 += bm.sum(0); s2 += (bm**2).sum(0); ntr += m.sum()
mu = s1/ntr; sd = np.sqrt(np.maximum(s2/ntr-mu**2,0))+1e-6
for st in range(0, len(X), B):
    blk = X[st:st+B]; blk -= mu; blk /= sd; np.clip(blk,-10,10,out=blk)
print('standardized, rss', rss(), flush=True)
np.save('/tmp/work/mlp60_mu.npy', mu); np.save('/tmp/work/mlp60_sd.npy', sd)
np.save('/tmp/work/mlp60_Xstd.npy', X)
np.save('/tmp/work/mlp60_y.npy', y); np.save('/tmp/work/mlp60_month.npy', month)
print('BUILD_DONE', flush=True)
