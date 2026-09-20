# X26: event/volume-synchronized microstructure (absolute, per-sample, NO cross-sectional).
# OFI (Cont-Kukanov-Stoikov) on L1/L2 1s snapshots + OFI acceleration, Roll implied spread,
# Amihud illiquidity, microprice tail stats, VPIN (volume-clock, K=10/20 buckets).
# IO pattern: build_x24/bx25 (pyarrow ziter, bincount accumulators). Runs on comp raw feathers.
# Source: OPEN_SOURCE_0920.md - last untried feature axis (volume/event-sync vs our time-sync stack).
import os, time
import numpy as np

DATA = os.environ.get('COMP', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = {'train': 1257637, 'test': 647896}
f64 = np.float64

def ziter(path, cols, elems=1<<19):
    import pyarrow as pa
    reader = pa.ipc.open_file(path)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        arrays = [b.column(c).to_numpy(zero_copy_only=False) for c in cols]
        n = len(arrays[0])
        for s in range(0, n, elems):
            yield tuple(a[s:s+elems] for a in arrays)

def market_pass(split):
    """OFI (L1+L2), OFI accel, Roll spread, Amihud, microprice tail."""
    ns = NS[split]; t0 = time.time()
    ofi_full = np.zeros(ns, f64); ofi_l300 = np.zeros(ns, f64); ofi_l60 = np.zeros(ns, f64)
    ofi_early = np.zeros(ns, f64); ofi2 = np.zeros(ns, f64); ofi_cnt = np.zeros(ns, f64)
    ofi_l2 = np.zeros(ns, f64); ofi_sgn = np.zeros(ns, f64)
    dm_sum = np.zeros(ns, f64); dm2_sum = np.zeros(ns, f64); dm_prev_prod = np.zeros(ns, f64)
    dm_cnt = np.zeros(ns, f64); amihud = np.zeros(ns, f64)
    mic_last = np.zeros(ns, f64); mic2 = np.zeros(ns, f64); mic_cnt = np.zeros(ns, f64)
    pbp = np.zeros(ns, f64); pbpv = np.zeros(ns, f64); pap = np.zeros(ns, f64); papv = np.zeros(ns, f64)
    pbp2 = np.zeros(ns, f64); pbpv2 = np.zeros(ns, f64); pap2 = np.zeros(ns, f64); papv2 = np.zeros(ns, f64)
    pmid = np.zeros(ns, f64); pdm = np.zeros(ns, f64); seen = np.zeros(ns, bool)
    for sid, sbp, a1, b1, av1, bv1, a2, b2, av2, bv2, txv in ziter(f'{DATA}/{split}/market.feather',
            ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1',
             'ask_price_2','bid_price_2','ask_volume_2','bid_volume_2','transaction_volume']):
        s64 = sid.astype(np.int64); tf = sbp.astype(f64)
        pa=a1.astype(f64); pb=b1.astype(f64); va=av1.astype(f64); vb=bv1.astype(f64)
        pa2=a2.astype(f64); pb2=b2.astype(f64); va2=av2.astype(f64); vb2=bv2.astype(f64)
        tx = txv.astype(f64)
        has = seen[s64]
        # CKS OFI on consecutive same-sample snapshots (rows sorted sid, sbp desc = chronological)
        ppb=pbp[s64]; ppbv=pbpv[s64]; ppa=pap[s64]; ppav=papv[s64]
        e_bid = vb*(pb>=ppb) - ppbv*(pb<=ppb)
        e_ask = va*(pa<=ppa) - ppav*(pa>=ppa)
        ofi = np.where(has, e_bid - e_ask, 0.0)
        # empty-book rows (price 0): zero the contribution
        valid = has & (pb>0) & (pa>0) & (ppb>0) & (ppa>0)
        ofi = np.where(valid, ofi, 0.0)
        ofi_full += np.bincount(s64, weights=ofi, minlength=ns)
        ofi2 += np.bincount(s64, weights=ofi*ofi, minlength=ns)
        ofi_cnt += np.bincount(s64[valid], minlength=ns)
        ofi_sgn += np.bincount(s64, weights=np.sign(ofi), minlength=ns)
        l300 = valid & (tf<300); l60 = valid & (tf<60); early = valid & (tf>=300)
        ofi_l300 += np.bincount(s64[l300], weights=ofi[l300], minlength=ns)
        ofi_l60 += np.bincount(s64[l60], weights=ofi[l60], minlength=ns)
        ofi_early += np.bincount(s64[early], weights=ofi[early], minlength=ns)
        # L2 OFI
        ppb2=pbp2[s64]; ppbv2=pbpv2[s64]; ppa2=pap2[s64]; ppav2=papv2[s64]
        e_bid2 = vb2*(pb2>=ppb2) - ppbv2*(pb2<=ppb2)
        e_ask2 = va2*(pa2<=ppa2) - ppav2*(pa2>=ppa2)
        valid2 = has & (pb2>0) & (pa2>0) & (ppb2>0) & (ppa2>0)
        ofi_l2 += np.bincount(s64, weights=np.where(valid2, e_bid2 - e_ask2, 0.0), minlength=ns)
        pbp[s64]=pb; pbpv[s64]=vb; pap[s64]=pa; papv[s64]=va
        pbp2[s64]=pb2; pbpv2[s64]=vb2; pap2[s64]=pa2; papv2[s64]=va2
        # mid changes: Roll spread + Amihud
        mid = (pa+pb)/2
        vmid = (pb>0)&(pa>0)
        pm = pmid[s64]
        dm = np.where(has & vmid & (pm>0), np.log(np.maximum(mid,1e-12)/np.maximum(pm,1e-12)), 0.0)
        dm_sum += np.bincount(s64, weights=dm, minlength=ns)
        dm2_sum += np.bincount(s64, weights=dm*dm, minlength=ns)
        dm_prev_prod += np.bincount(s64, weights=dm*pdm[s64], minlength=ns)
        dm_cnt += np.bincount(s64[has & vmid], minlength=ns)
        amihud += np.bincount(s64, weights=np.abs(dm)/np.maximum(tx,1.0), minlength=ns)
        pdm[s64] = dm
        pmid[s64] = np.where(vmid, mid, pm)
        seen[s64] = True
        # microprice tail: last value (min sbp seen = last row) + variance
        mic = np.where(vmid, (pa*vb + pb*va)/np.maximum(va+vb,1e-9)/np.maximum(mid,1e-9) - 1, 0.0)
        mic2 += np.bincount(s64, weights=mic*mic, minlength=ns)
        mic_cnt += np.bincount(s64[vmid], minlength=ns)
        upd = vmid & (tf <= 1.0)
        mic_last[s64[upd]] = mic[upd]
    print(split, 'X26 market pass', round(time.time()-t0), 's', flush=True)
    oc = np.maximum(ofi_cnt, 1.0); dc = np.maximum(dm_cnt, 1.0)
    ofi_mu = ofi_full/oc
    ofi_sd = np.sqrt(np.maximum(ofi2/oc - ofi_mu**2, 0))
    roll = 2*np.sqrt(np.maximum(-(dm_prev_prod/dc - (dm_sum/dc)*(dm_sum/dc)), 0))
    feats = [
        ofi_full/oc,                    # OFI mean per valid pair
        ofi_l300/oc,                    # OFI late 300s (scaled by full cnt: keeps magnitude meaning)
        ofi_l60/oc,
        ofi_early/oc,
        (ofi_early - ofi_l300)/oc,      # OFI acceleration (early - late)
        ofi_sd,                         # OFI volatility
        ofi_l2/oc,                      # L2 OFI
        ofi_sgn/oc,                     # OFI sign persistence
        roll,                           # Roll implied spread
        amihud/dc,                      # Amihud illiquidity
        mic_last,                       # microprice dev at window end
        np.sqrt(np.maximum(mic2/np.maximum(mic_cnt,1.0),0)),  # microprice dev RMS
    ]
    return feats

def tx_pass_vpin(split, K):
    """VPIN: two-pass volume-clock. Pass1 total vol; pass2 signed nets per volume bucket."""
    ns = NS[split]; t0 = time.time()
    tot = np.zeros(ns, f64)
    for sid, sbp, pr, v, side in ziter(f'{DATA}/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side']):
        tot += np.bincount(sid.astype(np.int64), weights=v.astype(f64), minlength=ns)
    bvol = np.maximum(tot/K, 1e-9)
    Nk = ns*(K+1)
    net = np.zeros(Nk, f64)
    cumv_off = np.zeros(ns, f64)  # cumulative volume per sample carried across chunks
    for sid, sbp, pr, v, side in ziter(f'{DATA}/{split}/transaction.feather',
            ['sample_id','seconds_before_predict','price','volume','side']):
        s64 = sid.astype(np.int64); vf = v.astype(f64)
        sgn = np.where(side==0, 1.0, -1.0)
        # volume-clock bucket by trade MIDPOINT (stable vs fp noise on exact bucket boundaries)
        cs = np.cumsum(vf)
        cs_before = cs - vf  # cumsum excluding current trade
        starts = np.concatenate(([0], np.nonzero(np.diff(s64)!=0)[0]+1))
        sample_start_cs = cs_before[starts]
        row_base = sample_start_cs[np.searchsorted(starts, np.arange(len(s64)), side='right')-1]
        run_cum = cs_before - row_base  # cum vol of this sample's earlier rows in this chunk
        cumv_mid = cumv_off[s64] + run_cum + vf/2
        bidx = np.minimum((cumv_mid/bvol[s64]).astype(np.int64), K)
        net += np.bincount(s64*(K+1)+bidx, weights=vf*sgn, minlength=Nk)
        # carry full cumulative volume (including current trade) to the next chunk
        end_rows = np.concatenate((starts[1:]-1, [len(s64)-1]))
        cumv_off[s64[end_rows]] = cumv_off[s64[end_rows]] + (run_cum + vf)[end_rows]
    net = net.reshape(ns, K+1)
    absnet = np.abs(net[:, :K])  # ignore overflow bucket K in mean
    nz = absnet.sum(axis=1) > 0
    vpin = np.where(nz, absnet.mean(axis=1), 0.0)/bvol
    print(split, f'X26 VPIN K={K}', round(time.time()-t0), 's', flush=True)
    return vpin

def build(split):
    ns = NS[split]
    feats = market_pass(split)
    feats.append(tx_pass_vpin(split, 10))
    feats.append(tx_pass_vpin(split, 20))
    X = np.stack([np.asarray(f_, f64) for f_ in feats], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(f'/kaggle/working/X26_{split}.npy', X)
    print(split, 'X26 done', X.shape, flush=True)

build('train'); build('test')
