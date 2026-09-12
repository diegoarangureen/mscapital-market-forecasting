import sys, numpy as np, gc
sys.path.insert(0,'/tmp/work')
from features2 import agg_market
split, ns = sys.argv[1], int(sys.argv[2])
import time; t=time.time()
out = agg_market(f'/tmp/mscapital/{split}/market.feather', ns)
print(f'{split} mk agg {time.time()-t:.0f}s', flush=True)
prev = dict(np.load(f'/tmp/work/part_{split}_txord.npz'))
prev.update(out); del out; gc.collect()
keys = sorted(prev)
X = np.memmap(f'/tmp/work/X2_{split}.npy.tmp', dtype=np.float32, mode='w+', shape=(ns, len(keys)))
for j, k in enumerate(keys):
    X[:, j] = prev[k].astype(np.float32)
    prev[k] = None
X.flush(); del X; gc.collect()
import os
os.rename(f'/tmp/work/X2_{split}.npy.tmp', f'/tmp/work/X2_{split}.raw')
np.save(f'/tmp/work/X2_{split}_keys.npy', np.array(keys))
# write proper npy header by loading memmap back via numpy save of view
Xr = np.memmap(f'/tmp/work/X2_{split}.raw', dtype=np.float32, mode='r', shape=(ns, len(keys)))
np.save(f'/tmp/work/X2_{split}.npy', Xr)
os.remove(f'/tmp/work/X2_{split}.raw')
print('saved', (ns, len(keys)), flush=True); print(keys, flush=True)