# build_seq v2: 120 steps x 8 channels, every market bar (no subsample), float16 memmap.
# channels: [dlogp_mid, dlogp_txp, log1p(txvol), log1p(txcnt), relspread, imb1, imb2, dmidvol]
import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter

NS = {'train':1257637, 'test':647896}
STEPS=120; CH=8
def build(split):
    ns = NS[split]
    out = np.memmap(f'/tmp/work/seq2_{split}.f16', dtype=np.float16, mode='w+', shape=(ns,STEPS,CH))
    cols = ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume','transaction_count',
            'ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
            'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2']
    t0=time.time()
    for ch in ziter(f'/tmp/mscapital/{split}/market.feather', cols, elems=1<<19):
        sid, sbp, ap, tv, tc, a1, b1, av1, bv1, a2, b2, av2, bv2 = ch
        n = len(sid)
        starts = np.r_[0, np.flatnonzero(np.diff(sid))+1]
        loc = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
        grp_start_rows = np.repeat(starts, np.diff(np.r_[starts, n]))
        idx_prev = np.maximum(np.arange(n)-1, grp_start_rows)
        mid = (a1+b1)/2.0
        prevmid = mid[idx_prev]; prevtx = ap[idx_prev]
        dlogp_mid = np.log(np.maximum(mid,1e-12)/np.maximum(prevmid,1e-12))
        dlogp_txp = np.log(np.maximum(ap,1e-12)/np.maximum(prevtx,1e-12))
        relsp = (a1-b1)/np.maximum(mid,1e-12)
        imb1 = (bv1-av1)/(bv1+av1+1); imb2 = (bv1+bv2-av1-av2)/(bv1+bv2+av1+av2+1)
        dmidvol = (av1+bv1) - (av1+bv1)[idx_prev]
        feats = np.stack([dlogp_mid, dlogp_txp, np.log1p(np.maximum(tv,0)), np.log1p(np.maximum(tc,0)),
                          relsp, imb1, imb2, np.log1p(np.abs(dmidvol))*np.sign(dmidvol)], axis=1)
        np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # keep LAST STEPS bars per sample: position from end
        glen = np.repeat(np.diff(np.r_[starts, n]), np.diff(np.r_[starts, n]))
        from_end = glen - 1 - loc
        ok = from_end < STEPS
        st2 = STEPS - 1 - from_end[ok]
        out[sid[ok], st2, :] = feats[ok].astype(np.float16)
    out.flush(); del out; gc.collect()
    print(split, 'seq2 built', round(time.time()-t0), 's', flush=True)

build('train'); build('test')
print('SEQ2_DONE', flush=True)
