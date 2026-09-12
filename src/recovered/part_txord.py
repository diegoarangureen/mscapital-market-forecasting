import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from features2 import agg_transaction, agg_order
split, ns = sys.argv[1], int(sys.argv[2])
base = f'/tmp/mscapital/{split}'
out = {}
t=time.time(); out.update(agg_transaction(f'{base}/transaction.feather', ns)); print(f'{split} tx {time.time()-t:.0f}s', flush=True); gc.collect()
t=time.time(); out.update(agg_order(f'{base}/order.feather', ns)); print(f'{split} ord {time.time()-t:.0f}s', flush=True); gc.collect()
np.savez(f'/tmp/work/part_{split}_txord.npz', **out)
print('saved partial', len(out), flush=True)