import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from features2 import agg_market
split, ns = sys.argv[1], int(sys.argv[2])
base = f'/tmp/mscapital/{split}'
t=time.time(); out = agg_market(f'{base}/market.feather', ns); print(f'{split} mk {time.time()-t:.0f}s', flush=True)
prev = dict(np.load(f'/tmp/work/part_{split}_txord.npz'))
prev.update(out)
keys = sorted(prev)
X = np.stack([prev[k] for k in keys], axis=1).astype(np.float32)
np.save(f'/tmp/work/X2_{split}.npy', X)
np.save(f'/tmp/work/X2_{split}_keys.npy', np.array(keys))
print('saved', X.shape, flush=True); print(keys, flush=True)