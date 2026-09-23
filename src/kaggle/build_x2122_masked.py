# X29: quantify price==0 (empty book levels, senanuretin finding) and build MASKED X21/X22
# variants (train only) kernel-side from the raw competition feather streams.
# Masking: price-derived L1 terms (mid, microprice, spread) use only rows with a1>0 & b1>0;
# X22 slope terms additionally require a2>0 / b2>0 (own counts). Volume-based imb and all
# tx/order features are unchanged. Empty-mask buckets -> 0.0 (trainer applies nan_to_num anyway).
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ['MALLOC_ARENA_MAX'] = '1'
import numpy as np
BASE = os.environ.get('BASE', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
NS = int(os.environ.get('NS', '1257637'))
SPLIT = os.environ.get('SPLIT', 'train')
OUT = '/kaggle/working'

import pyarrow as pa
def ziter(path, cols, elems=1<<19):
    reader = pa.ipc.open_file(path)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        arrays = [b.column(c).to_numpy(zero_copy_only=False) for c in cols]
        n = len(arrays[0])
        for s in range(0, n, elems):
            yield tuple(a[s:s+elems] for a in arrays)

# ---------- 1) quantify price==0 ----------
t0 = time.time()
tot = 0; za1 = 0; zb1 = 0; za2 = 0; zb2 = 0
for a1, b1, a2, b2 in ziter(f'{BASE}/{SPLIT}/market.feather',
        ['ask_price_1','bid_price_1','ask_price_2','bid_price_2']):
    tot += len(a1); za1 += int((a1==0).sum()); zb1 += int((b1==0).sum())
    za2 += int((a2==0).sum()); zb2 += int((b2==0).sum())
print(f'QUANT market rows={tot} ask1==0 {za1} ({za1/tot:.4%}) bid1==0 {zb1} ({zb1/tot:.4%}) '
      f'ask2==0 {za2} ({za2/tot:.4%}) bid2==0 {zb2} ({zb2/tot:.4%})', flush=True)
tot_o = 0; zpo = 0
for pr, in ziter(f'{BASE}/{SPLIT}/order.feather', ['price']):
    tot_o += len(pr); zpo += int((pr==0).sum())
print(f'QUANT order rows={tot_o} price==0 {zpo} ({zpo/tot_o:.4%})', flush=True)
tot_t = 0; zpt = 0
for pr, in ziter(f'{BASE}/{SPLIT}/transaction.feather', ['price']):
    tot_t += len(pr); zpt += int((pr==0).sum())
print(f'QUANT transaction rows={tot_t} price==0 {zpt} ({zpt/tot_t:.4%})', flush=True)
print('quantify done', round(time.time()-t0), 's', flush=True)

f64 = np.float64

# ---------- 2) masked X21 (3 early buckets [0,10),[10,30),[30,60)) ----------
t0 = time.time(); NB = 3; N = NS*NB
edges = np.array([10.0, 30.0])
n   = np.zeros(N, f64); nv = np.zeros(N, f64)
smicro = np.zeros(N, f64); sspread = np.zeros(N, f64); simb = np.zeros(N, f64); smid_f = np.zeros(N, f64)
for sid, sbp, a1, b1, av1, bv1 in ziter(f'{BASE}/{SPLIT}/market.feather',
        ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1']):
    b = np.searchsorted(edges, sbp.astype(f64))
    lin = sid.astype(np.int64)*NB + b
    a1f = a1.astype(f64); b1f = b1.astype(f64)
    av = av1.astype(f64); bv = bv1.astype(f64)
    valid = (a1f > 0) & (b1f > 0)
    vf_ = valid.astype(f64)
    mid = np.where(valid, (a1f+b1f)/2, 0.0)
    micro = np.where(valid, (a1f*bv + b1f*av)/np.maximum(av+bv, 1e-9), 0.0)
    spread = np.where(valid, (a1f-b1f)/np.maximum((a1f+b1f)/2, 1e-9), 0.0)
    imb = (bv-av)/np.maximum(av+bv, 1e-9)
    n += np.bincount(lin, minlength=N)
    nv += np.bincount(lin, weights=vf_, minlength=N)
    smicro += np.bincount(lin, weights=micro, minlength=N)
    sspread += np.bincount(lin, weights=spread, minlength=N)
    simb += np.bincount(lin, weights=imb, minlength=N)
    smid_f += np.bincount(lin, weights=mid, minlength=N)
print('X21m market pass', round(time.time()-t0), 's', flush=True)
tn = np.zeros(N, f64); tsvol = np.zeros(N, f64); tsamt = np.zeros(N, f64); ttot = np.zeros(N, f64)
for sid, sbp, pr, v, side in ziter(f'{BASE}/{SPLIT}/transaction.feather',
        ['sample_id','seconds_before_predict','price','volume','side']):
    b = np.searchsorted(edges, sbp.astype(f64))
    lin = sid.astype(np.int64)*NB + b
    vf = v.astype(f64); pf = pr.astype(f64)
    sgn = np.where(side==0, 1.0, -1.0)
    tn += np.bincount(lin, minlength=N)
    tsvol += np.bincount(lin, weights=vf*sgn, minlength=N)
    tsamt += np.bincount(lin, weights=pf*vf*sgn, minlength=N)
    ttot += np.bincount(lin, weights=vf, minlength=N)
print('X21m tx pass', round(time.time()-t0), 's', flush=True)
onb = np.zeros(N, f64); ons = np.zeros(N, f64); ocb = np.zeros(N, f64); ocs = np.zeros(N, f64)
for sid, sbp, pr, v, side, act in ziter(f'{BASE}/{SPLIT}/order.feather',
        ['sample_id','seconds_before_predict','price','volume','side','order_action']):
    b = np.searchsorted(edges, sbp.astype(f64))
    lin = sid.astype(np.int64)*NB + b
    vf = v.astype(f64)
    is_new = (act==0); buy = (side==0)
    onb += np.bincount(lin, weights=np.where(is_new & buy, vf, 0.0), minlength=N)
    ons += np.bincount(lin, weights=np.where(is_new & ~buy, vf, 0.0), minlength=N)
    ocb += np.bincount(lin, weights=np.where(~is_new & buy, vf, 0.0), minlength=N)
    ocs += np.bincount(lin, weights=np.where(~is_new & ~buy, vf, 0.0), minlength=N)
print('X21m order pass', round(time.time()-t0), 's', flush=True)
names = []; cols = []
for bi in range(NB):
    sl = np.s_[bi::NB]
    nn = np.maximum(n[sl], 1); nvv = np.maximum(nv[sl], 1); tt = np.maximum(ttot[sl], 1e-9)
    midm = smid_f[sl]/nvv  # masked mid mean
    feats = {
      f'x21_m{bi}_micro_dev': np.where(nv[sl]>0, smicro[sl]/np.maximum(smid_f[sl],1e-9)-1, 0.0),
      f'x21_m{bi}_spread': sspread[sl]/nvv,
      f'x21_m{bi}_imb1': simb[sl]/nn,
      f'x21_t{bi}_signvol': tsvol[sl],
      f'x21_t{bi}_signvol_frac': tsvol[sl]/tt,
      f'x21_t{bi}_signamt': tsamt[sl]/np.maximum(midm,1e-9),
      f'x21_t{bi}_vol': ttot[sl],
      f'x21_o{bi}_newpress': (onb[sl]-ons[sl]),
      f'x21_o{bi}_canimb': (ocs[sl]-ocb[sl])/np.maximum(ocs[sl]+ocb[sl],1e-9),
      f'x21_o{bi}_netplace': (onb[sl]+ons[sl])-(ocb[sl]+ocs[sl]),
    }
    for k, vv in feats.items():
        names.append(k); cols.append(vv.astype(np.float32))
X = np.stack(cols, axis=1)
np.save(f'{OUT}/X21m_{SPLIT}.npy', X); np.save(f'{OUT}/X21m_{SPLIT}_names.npy', np.array(names))
print('X21m saved', X.shape, round(time.time()-t0), 's', flush=True)
del X, cols, n, nv, smicro, sspread, simb, smid_f, tn, tsvol, tsamt, ttot, onb, ons, ocb, ocs

# ---------- 3) masked X22 (early buckets + late ref) ----------
t0 = time.time()
N5 = NS*5
edges4 = np.array([10.0, 30.0, 60.0, 300.0])
nm = np.zeros(N5, f64); nv1 = np.zeros(N5, f64); nva = np.zeros(N5, f64); nvb = np.zeros(N5, f64)
ssa = np.zeros(N5, f64); ssb = np.zeros(N5, f64)
simb = np.zeros(N5, f64); sspread = np.zeros(N5, f64); smicro = np.zeros(N5, f64); smid = np.zeros(N5, f64)
for sid, sbp, a1, b1, av1, bv1, a2, b2 in ziter(f'{BASE}/{SPLIT}/market.feather',
        ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1','ask_price_2','bid_price_2']):
    b = np.searchsorted(edges4, sbp.astype(f64))
    lin = sid.astype(np.int64)*5 + b
    a1f=a1.astype(f64); b1f=b1.astype(f64); a2f=a2.astype(f64); b2f=b2.astype(f64)
    av=av1.astype(f64); bv=bv1.astype(f64)
    v1 = (a1f>0)&(b1f>0); va = (a1f>0)&(a2f>0); vb = (b1f>0)&(b2f>0)
    m1 = np.where(v1, np.maximum((a1f+b1f)/2, 1e-9), 1.0)
    nm += np.bincount(lin, minlength=N5)
    nv1 += np.bincount(lin, weights=v1.astype(f64), minlength=N5)
    nva += np.bincount(lin, weights=va.astype(f64), minlength=N5)
    nvb += np.bincount(lin, weights=vb.astype(f64), minlength=N5)
    ssa += np.bincount(lin, weights=np.where(va, (a2f-a1f)/m1, 0.0), minlength=N5)
    ssb += np.bincount(lin, weights=np.where(vb, (b1f-b2f)/m1, 0.0), minlength=N5)
    simb += np.bincount(lin, weights=(bv-av)/np.maximum(av+bv,1e-9), minlength=N5)
    sspread += np.bincount(lin, weights=np.where(v1, (a1f-b1f)/m1, 0.0), minlength=N5)
    smicro += np.bincount(lin, weights=np.where(v1, (a1f*bv+b1f*av)/np.maximum(av+bv,1e-9), 0.0), minlength=N5)
    smid += np.bincount(lin, weights=np.where(v1, (a1f+b1f)/2, 0.0), minlength=N5)
print('X22m market pass', round(time.time()-t0), 's', flush=True)
tsv = np.zeros(N5, f64)
for sid, sbp, pr, v, side in ziter(f'{BASE}/{SPLIT}/transaction.feather',
        ['sample_id','seconds_before_predict','price','volume','side']):
    b = np.searchsorted(edges4, sbp.astype(f64))
    lin = sid.astype(np.int64)*5 + b
    sgn = np.where(side==0,1.0,-1.0)
    tsv += np.bincount(lin, weights=v.astype(f64)*sgn, minlength=N5)
print('X22m tx pass', round(time.time()-t0), 's', flush=True)
onb = np.zeros(N5, f64); ons = np.zeros(N5, f64); ocb = np.zeros(N5, f64); ocs = np.zeros(N5, f64)
for sid, sbp, pr, v, side, act in ziter(f'{BASE}/{SPLIT}/order.feather',
        ['sample_id','seconds_before_predict','price','volume','side','order_action']):
    b = np.searchsorted(edges4, sbp.astype(f64))
    lin = sid.astype(np.int64)*5 + b
    vf = v.astype(f64); is_new = (act==0); buy = (side==0)
    onb += np.bincount(lin, weights=np.where(is_new&buy, vf,0.0), minlength=N5)
    ons += np.bincount(lin, weights=np.where(is_new&~buy, vf,0.0), minlength=N5)
    ocb += np.bincount(lin, weights=np.where(~is_new&buy, vf,0.0), minlength=N5)
    ocs += np.bincount(lin, weights=np.where(~is_new&~buy, vf,0.0), minlength=N5)
print('X22m order pass', round(time.time()-t0), 's', flush=True)
names=[]; cols=[]
nn = np.maximum(nm,1); nv1m = np.maximum(nv1,1); nvam = np.maximum(nva,1); nvbm = np.maximum(nvb,1)
imb = simb/nn
spread = np.where(nv1>0, sspread/nv1m, 0.0)
microdev = np.where(nv1>0, smicro/np.maximum(smid,1e-9)-1, 0.0)
slopea = np.where(nva>0, ssa/nvam, 0.0); slopeb = np.where(nvb>0, ssb/nvbm, 0.0)
E = [0,1,2]; LATE = 4
for bi in E:
    sl = np.s_[bi::5]
    names += [f'x22_sla{bi}', f'x22_slb{bi}']; cols += [slopea[sl].astype(np.float32), slopeb[sl].astype(np.float32)]
    names.append(f'x22_sv{bi}'); cols.append(tsv[sl].astype(np.float32))
    names.append(f'x22_svXimb{bi}'); cols.append((tsv[sl]*imb[sl]).astype(np.float32))
    names.append(f'x22_cnrb{bi}'); cols.append((ocb[sl]/np.maximum(onb[sl],1e-9)).astype(np.float32))
    names.append(f'x22_cnrs{bi}'); cols.append((ocs[sl]/np.maximum(ons[sl],1e-9)).astype(np.float32))
    names.append(f'x22_net{bi}'); cols.append(((onb[sl]-ocb[sl])-(ons[sl]-ocs[sl])).astype(np.float32))
def S(bi): return np.s_[bi::5]
names.append('x22_d_sv_01');  cols.append((tsv[S(0)]-tsv[S(1)]).astype(np.float32))
names.append('x22_d_sv_12');  cols.append((tsv[S(1)]-tsv[S(2)]).astype(np.float32))
names.append('x22_d_imb_02'); cols.append((imb[S(0)]-imb[S(2)]).astype(np.float32))
names.append('x22_d_imb_0L'); cols.append((imb[S(0)]-imb[S(LATE)]).astype(np.float32))
names.append('x22_d_spr_0L'); cols.append((spread[S(0)]-spread[S(LATE)]).astype(np.float32))
names.append('x22_d_mic_0L'); cols.append((microdev[S(0)]-microdev[S(LATE)]).astype(np.float32))
X = np.stack(cols, axis=1)
np.save(f'{OUT}/X22m_{SPLIT}.npy', X); np.save(f'{OUT}/X22m_{SPLIT}_names.npy', np.array(names))
print('X22m saved', X.shape, round(time.time()-t0), 's', flush=True)
print('ALL DONE', flush=True)
