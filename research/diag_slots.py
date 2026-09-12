import sys, numpy as np, time
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter
import pyarrow.feather as feather

month_target = int(sys.argv[1]) if len(sys.argv)>1 else 61
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id')
sids = lab.sample_id.values[lab.month.values==month_target]
sid_min, sid_max = sids.min(), sids.max()
ns = len(sids)
print('month', month_target, 'nsamples', ns, 'sid range', sid_min, sid_max, flush=True)

MAXB = 400
px = np.full((ns, MAXB), np.nan, np.float32)
vl = np.zeros((ns, MAXB), np.float32)
sbparr = np.full((ns, MAXB), np.nan, np.float32)
cnt = np.zeros(ns, np.int64)
lut = np.full(sid_max+1, -1, np.int64); lut[sids] = np.arange(ns)

t0=time.time()
for sid, sbp, ap, tv in ziter('/tmp/mscapital/train/market.feather',
        ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume']):
    if sid[0] > sid_max: break
    m = (sid>=sid_min)&(sid<=sid_max)
    if not m.any():
        continue
    s=sid[m]; a=ap[m]; v=tv[m]; b=sbp[m]
    # per-row position within sample: rows sorted by sid -> contiguous blocks
    first_idx = np.maximum.accumulate(np.arange(len(s)) * (np.r_[True, s[1:]!=s[:-1]]))
    pos = (np.arange(len(s)) - first_idx) + cnt[lut[s]]
    idx = lut[s]
    ok = pos<MAXB
    px[idx[ok], pos[ok]] = a[ok]
    vl[idx[ok], pos[ok]] = v[ok]
    sbparr[idx[ok], pos[ok]] = b[ok]
    cnt[idx] = pos+1
print('collect', round(time.time()-t0), 's', flush=True)
print('bars/sample distribution:', np.bincount(np.minimum(cnt,20))[:21], flush=True)
np.savez('/tmp/work/diag_slots.npz', px=px, vl=vl, sbp=sbparr, cnt=cnt, sids=sids)
print('saved', flush=True)
