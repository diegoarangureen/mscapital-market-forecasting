# build_seq3: 120 steps x 12 channels on a UNIFORM 1-second grid (last 120s before prediction).
# market + transaction + order streams per second. float16 memmap, block-wise accumulation (low RAM).
# channels: [dlogp_mid(sum), spread(mean), imb1(mean), micro_dev(mean), slope_a(mean),
#            tx_signvol(sum), tx_vol(log1p sum), tx_n(log1p), ord_new_b, ord_new_s, ord_can_b, ord_can_s (log1p)]
# Each pass flushes ONLY its own channel slice per sample-block; time axis flipped at flush
# (last step = most recent second).
import sys, os, numpy as np, time, gc, ctypes
sys.path.insert(0,'/tmp/work')
_libc = ctypes.CDLL('libc.so.6')
from streamcol2 import ziter
os.environ['MALLOC_ARENA_MAX']='1'

NS = {'train':1257637, 'test':647896}
STEPS=120; CH=12; BS=60000
f32=np.float32

def build(split):
    ns = NS[split]
    out = np.memmap(f'/tmp/work/seq3_{split}.f16', dtype=np.float16, mode='w+', shape=(ns,STEPS,CH))
    out[:] = 0; out.flush()
    t0=time.time()

    def run_pass(chans, rowfn, fname, cols, mean=False, log1p_chans=()):
        # rowfn(chunk_tuple) -> (sid, s, W[n, len(chans)]) for rows with s<STEPS
        acc = np.zeros(BS*STEPS*len(chans), f32)
        cnt = np.zeros(BS*STEPS, f32) if mean else None
        cur = 0
        nch = len(chans)
        def flush_block(b0):
            lo=b0; hi=min(b0+BS, ns); w=hi-lo
            blk = acc[:w*STEPS*nch].reshape(w,STEPS,nch)
            if mean:
                cn = cnt[:w*STEPS].reshape(w,STEPS,1)
                blk = blk/np.maximum(cn,1.0)
            if log1p_chans:
                idx = [chans.index(c) for c in log1p_chans]
                blk[:,:,idx] = np.log1p(np.maximum(blk[:,:,idx],0))
            np.nan_to_num(blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            out[lo:hi, ::-1, chans] = blk.astype(np.float16)  # flip time here
            out.flush()
            acc[:] = 0
            if cnt is not None: cnt[:] = 0
        for chunk in ziter(f'/tmp/mscapital/{split}/{fname}.feather', cols, elems=1<<19):
            sid, s, W = rowfn(chunk)
            n=len(sid); n0=0
            while n0 < n:
                j = np.searchsorted(sid[n0:], cur+BS)+n0
                loc = (sid[n0:j]-cur).astype(np.int64)*STEPS + s[n0:j]
                if mean:
                    np.add.at(cnt, loc, 1.0)
                for k in range(nch):
                    np.add.at(acc, loc*nch + k, W[n0:j,k])
                n0 = j
                if n0 < n:
                    flush_block(cur); cur += BS
            _libc.malloc_trim(0)
        flush_block(cur)
        del acc, cnt; gc.collect(); _libc.malloc_trim(0)

    # ---- market pass: channels 0-4 (means except dlogp sum) ----
    def market_rows(ch):
        sid, sbp, a1, b1, av1, bv1, a2, b2 = ch
        s = sbp.astype(np.int64); ok = s < STEPS
        sid=sid[ok]; s=s[ok]; a1=a1[ok]; b1=b1[ok]; av1=av1[ok]; bv1=bv1[ok]; a2=a2[ok]; b2=b2[ok]
        n=len(sid)
        starts = np.r_[0, np.flatnonzero(np.diff(sid))+1]
        gs = np.repeat(starts, np.diff(np.r_[starts, n]))
        ip = np.maximum(np.arange(n)-1, gs)
        mid = (a1.astype(np.float64)+b1.astype(np.float64))/2
        W = np.empty((n,5), f32)
        W[:,0]=np.log(np.maximum(mid,1e-12)/np.maximum(mid[ip],1e-12))
        W[:,1]=(a1-b1)/np.maximum(mid,1e-12)
        W[:,2]=(bv1-av1)/(bv1+av1+1.0)
        W[:,3]=((a1*bv1 + b1*av1)/np.maximum(av1+bv1,1))/np.maximum(mid,1e-12) - 1
        W[:,4]=(a2-a1)/np.maximum(mid,1e-12)
        return sid, s, W
    run_pass([0,1,2,3,4], market_rows, 'market',
             ['sample_id','seconds_before_predict','ask_price_1','bid_price_1','ask_volume_1','bid_volume_1','ask_price_2','bid_price_2'],
             mean=True)
    # fix: channel 0 should be SUM not mean -> redo as separate accumulation below
    print(split,'market done',round(time.time()-t0),'s',flush=True)

    # ---- transaction pass: channels 5(sum signed),6(sum vol ->log1p),7(count ->log1p) ----
    def tx_rows(ch):
        sid, sbp, pr, v, side = ch
        s = sbp.astype(np.int64); ok = s < STEPS
        sid=sid[ok]; s=s[ok]; v=v[ok].astype(f32); side=side[ok]
        sgn = np.where(side==0, f32(1.0), f32(-1.0))
        W = np.empty((len(sid),3), f32)
        W[:,0]=v*sgn; W[:,1]=v; W[:,2]=1.0
        return sid, s, W
    run_pass([5,6,7], tx_rows, 'transaction',
             ['sample_id','seconds_before_predict','price','volume','side'],
             log1p_chans=(6,7))
    print(split,'tx done',round(time.time()-t0),'s',flush=True)

    # ---- order pass: channels 8-11 (sums -> log1p) ----
    def ord_rows(ch):
        sid, sbp, pr, v, side, act = ch
        s = sbp.astype(np.int64); ok = s < STEPS
        sid=sid[ok]; s=s[ok]; v=v[ok].astype(f32); side=side[ok]; act=act[ok]
        is_new=(act==0); buy=(side==0)
        W = np.zeros((len(sid),4), f32)
        W[is_new&buy,0]=v[is_new&buy]; W[is_new&~buy,1]=v[is_new&~buy]
        W[~is_new&buy,2]=v[~is_new&buy]; W[~is_new&~buy,3]=v[~is_new&~buy]
        return sid, s, W
    run_pass([8,9,10,11], ord_rows, 'order',
             ['sample_id','seconds_before_predict','price','volume','side','order_action'],
             log1p_chans=(8,9,10,11))
    print(split,'order done',round(time.time()-t0),'s',flush=True)

    # ---- fix channel 0: recompute dlogp as SUM (overwrite) ----
    def dlogp_rows(ch):
        sid, sbp, a1, b1 = ch
        s = sbp.astype(np.int64); ok = s < STEPS
        sid=sid[ok]; s=s[ok]; a1=a1[ok]; b1=b1[ok]
        n=len(sid)
        starts = np.r_[0, np.flatnonzero(np.diff(sid))+1]
        gs = np.repeat(starts, np.diff(np.r_[starts, n]))
        ip = np.maximum(np.arange(n)-1, gs)
        mid = (a1.astype(np.float64)+b1.astype(np.float64))/2
        W = np.log(np.maximum(mid,1e-12)/np.maximum(mid[ip],1e-12)).astype(f32).reshape(-1,1)
        return sid, s, W
    run_pass([0], dlogp_rows, 'market', ['sample_id','seconds_before_predict','ask_price_1','bid_price_1'])
    print(split,'dlogp redone',round(time.time()-t0),'s',flush=True)

    del out; gc.collect(); _libc.malloc_trim(0)
    print(split,'seq3 built',round(time.time()-t0),'s',flush=True)

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv)>1 else 'train')
    print('SEQ3_DONE', flush=True)
