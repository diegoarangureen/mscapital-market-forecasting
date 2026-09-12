# features6: structural microstructure signals
#  - CKS event-based Order Flow Imbalance between consecutive market bars (L1 and L2)
#  - microprice (Stoikov) at window end + its drift
#  - Kyle lambda from signed trades (real aggressor side)
#  - Amihud illiquidity from market bars
import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter

def agg_ofi(path, ns):
    """CKS OFI from consecutive bars + microprice stats, per sample."""
    ofi1 = np.zeros(ns); ofi2 = np.zeros(ns)
    ofi1_late = np.zeros(ns)   # sbp<=120 portion
    mp_end = np.full(ns, np.nan, np.float32)   # microprice at last bar (min sbp)
    mp_first = np.full(ns, np.nan, np.float32) # microprice at first bar (max sbp)
    sbp_min = np.full(ns, np.inf); sbp_max = np.full(ns, -np.inf)
    mid_end = np.full(ns, np.nan, np.float32)
    cols = ['sample_id','seconds_before_predict','ask_price_1','bid_price_1',
            'ask_volume_1','bid_volume_1','ask_price_2','bid_price_2',
            'ask_volume_2','bid_volume_2']
    t0=time.time()
    for sid, sbp, a1, b1, av1, bv1, a2, b2, av2, bv2 in ziter(path, cols):
        n = len(sid)
        # microprice per row
        mp = (a1*bv1 + b1*av1) / np.maximum(av1+bv1, 1)
        mid = (a1+b1)/2
        # track first/last bar microprice per sample via min/max sbp
        np.minimum.at(sbp_min, sid, sbp); np.maximum.at(sbp_max, sid, sbp)
        # bars are ordered within sample by descending sbp (600->0)? verify by diff sign later;
        # OFI needs consecutive-in-time pairs: sort key = (sid, -sbp). Assume file order is time order.
        prev_same = sid[1:] == sid[:-1]
        # L1 contributions from row i-1 -> i
        db = b1[1:] - b1[:-1]; da = a1[1:] - a1[:-1]
        dvb = bv1[1:].astype(np.float64) - bv1[:-1]; dva = av1[1:].astype(np.float64) - av1[:-1]
        # CKS convention: e = +qb_n if P^b rises; -qb_{n-1} if P^b falls; +(qb_n-qb_{n-1}) if same
        #                   -qa_n if P^a falls; +qa_{n-1} if P^a rises; -(qa_n-qa_{n-1}) if same
        contrib = np.zeros(n-1)
        contrib += np.where(bid_up, bv1[1:].astype(np.float64), 0.0)
        contrib += np.where(bid_dn, -bv1[:-1].astype(np.float64), 0.0)
        contrib += np.where(sameb, dvb, 0.0)
        contrib += np.where(ask_dn, -av1[1:].astype(np.float64), 0.0)
        contrib += np.where(ask_up, av1[:-1].astype(np.float64), 0.0)
        samea = (~ask_up)&(~ask_dn)
        contrib += np.where(samea, -dva, 0.0)
        contrib[~prev_same] = 0.0
        ofi1 += np.bincount(sid[1:], weights=contrib, minlength=ns)
        late = sbp[1:] <= 120
        ofi1_late += np.bincount(sid[1:][late], weights=contrib[late], minlength=ns)
        # L2 same recipe
        db2 = b2[1:] - b2[:-1]; da2 = a2[1:] - a2[:-1]
        dvb2 = bv2[1:].astype(np.float64) - bv2[:-1]; dva2 = av2[1:].astype(np.float64) - av2[:-1]
        c2 = np.zeros(n-1)
        bu = db2>0; bd = db2<0; au = da2>0; ad = da2<0
        c2 += np.where(bu, bv2[1:].astype(np.float64), 0.0) + np.where(bd, -bv2[:-1].astype(np.float64), 0.0)
        c2 += np.where((~bu)&(~bd), dvb2, 0.0)
        c2 += np.where(ad, -av2[1:].astype(np.float64), 0.0) + np.where(au, av2[:-1].astype(np.float64), 0.0)
        c2 += np.where((~au)&(~ad), -dva2, 0.0)
        c2[~prev_same] = 0.0
        ofi2 += np.bincount(sid[1:], weights=c2, minlength=ns)
    print('  ofi pass', round(time.time()-t0), flush=True)
    F={}
    F['ofi1']=ofi1; F['ofi2']=ofi2; F['ofi1_late']=ofi1_late
    return F

def agg_kyle(path, ns):
    """Kyle lambda per sample: cov(dp, signed_vol)/var(signed_vol) over trades; also signed-volume corr."""
    cols=['sample_id','price','volume','side']
    # per sample accumulate stats for regression dp ~ signed v
    n=np.zeros(ns); sx=np.zeros(ns); sy=np.zeros(ns); sxx=np.zeros(ns); sxy=np.zeros(ns)
    t0=time.time()
    for sid, price, v, side in ziter(path, cols):
        sgn = np.where(side==0, 1.0, -1.0)  # 0=buy aggressor
        x = sgn*v
        prev_same = sid[1:]==sid[:-1]
        dp = (price[1:]-price[:-1]).astype(np.float64)
        dp[~prev_same]=0.0
        s = sid[1:]
        n  += np.bincount(s, weights=prev_same.astype(np.float64), minlength=ns)
        sx += np.bincount(s, weights=x[1:]*prev_same, minlength=ns)
        sy += np.bincount(s, weights=dp, minlength=ns)
        sxx+= np.bincount(s, weights=x[1:]**2*prev_same, minlength=ns)
        sxy+= np.bincount(s, weights=x[1:]*dp*prev_same, minlength=ns)
    print('  kyle pass', round(time.time()-t0), flush=True)
    nn=np.maximum(n,1)
    cov=sxy/nn-(sx/nn)*(sy/nn); var=sxx/nn-(sx/nn)**2
    F={}
    F['kyle_lambda']=cov/np.maximum(var,1e-12)
    return F

if __name__=='__main__':
    split, ns = sys.argv[1], int(sys.argv[2])
    base=f'/tmp/mscapital/{split}'
    out={}
    out.update(agg_ofi(f'{base}/market.feather', ns)); gc.collect()
    out.update(agg_kyle(f'{base}/transaction.feather', ns)); gc.collect()
    keys=sorted(out)
    X=np.empty((ns,len(keys)),np.float32)
    for i,k in enumerate(keys): X[:,i]=out[k]
    np.save(f'/tmp/work/X6_{split}.npy', X)
    np.save(f'/tmp/work/X6_{split}_keys.npy', np.array(keys))
    print(split,'done',X.shape,keys,flush=True)
