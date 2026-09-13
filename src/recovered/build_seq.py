# Build per-sample bar sequences: (n, 60, 6) float16 memmap from market.feather
# channels: [dlogp, log1p(vol), log1p(cnt), relspread, imb1, imb2] ; every 3rd bar
import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter

NS = {'train':1257637, 'test':647896}
STEPS=60; CH=6
def build(split):
    ns = NS[split]
    out = np.memmap(f'/tmp/work/seq_{split}.f16', dtype=np.float16, mode='w+', shape=(ns,STEPS,CH))
    cnt = np.zeros(ns, np.int32)
    cols = ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume','transaction_count',
            'ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
            'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2']
    t0=time.time()
    for ch in ziter(f'/tmp/mscapital/{split}/market.feather', cols, elems=1<<19):
        sid, sbp, ap, tv, tc, a1, b1, av1, bv1, a2, b2, av2, bv2 = ch
        n = len(sid)
        # local index within sample (rows sorted by sid)
        starts = np.r_[0, np.flatnonzero(np.diff(sid))+1]
        # local position = arange within group
        loc = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
        # previous price per row (within-sample); first bar uses itself
        prevp = ap.copy()
        grp_start_rows = np.repeat(starts, np.diff(np.r_[starts, n]))
        idx_prev = np.maximum(np.arange(n)-1, grp_start_rows)
        prevp = ap[idx_prev]
        dlogp = np.log(np.maximum(ap,1e-12)/np.maximum(prevp,1e-12))
        mid = (a1+b1)/2.0
        relsp = (a1-b1)/np.maximum(mid,1e-12)
        imb1 = (bv1-av1)/(bv1+av1+1); imb2 = (bv1+bv2-av1-av2)/(bv1+bv2+av1+av2+1)
        feats = np.stack([dlogp, np.log1p(np.maximum(tv,0)), np.log1p(np.maximum(tc,0)), relsp, imb1, imb2], axis=1)
        np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        m0 = (loc % 3)==0
        step = loc[m0]//3
        ok = step < STEPS
        s2 = sid[m0][ok]; st2 = step[ok]
        out[s2, st2, :] = feats[m0][ok].astype(np.float16)
    out.flush(); del out; gc.collect()
    print(split, 'seq built', round(time.time()-t0), 's', flush=True)

build('train'); build('test')
print('SEQ_DONE', flush=True)
