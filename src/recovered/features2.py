import sys, numpy as np, time, gc
sys.path.insert(0,'/tmp/work')
from streamcol2 import ziter

HORIZONS = (15, 30, 60, 120, 300)

def agg_transaction(path, ns):
    F = {}
    n   = np.zeros(ns); vol = np.zeros(ns)
    bvol= np.zeros(ns); svol= np.zeros(ns)
    pv  = np.zeros(ns); p2v = np.zeros(ns)
    maxv= np.zeros(ns); sbpv= np.zeros(ns)
    mn  = np.full(ns, np.inf, np.float32); mx = np.full(ns, -np.inf, np.float32)
    W = (15, 30)
    wvol = {w: np.zeros(ns) for w in W}; wbv = {w: np.zeros(ns) for w in W}; wsv = {w: np.zeros(ns) for w in W}
    t0=time.time()
    for sid, sbp, price, v, side in ziter(path, ['sample_id','seconds_before_predict','price','volume','side']):
        n   += np.bincount(sid, minlength=ns)
        vol += np.bincount(sid, weights=v, minlength=ns)
        buy = (side==0)
        bvol+= np.bincount(sid, weights=v*buy, minlength=ns)
        svol+= np.bincount(sid, weights=v*(~buy), minlength=ns)
        p64 = price.astype(np.float64)
        pv  += np.bincount(sid, weights=p64*v, minlength=ns)
        p2v += np.bincount(sid, weights=p64*p64*v, minlength=ns)
        sbpv+= np.bincount(sid, weights=sbp.astype(np.float64)*v, minlength=ns)
        np.maximum.at(maxv, sid, v)
        np.minimum.at(mn, sid, sbp); np.maximum.at(mx, sid, sbp)
        for w in W:
            msk = sbp <= w
            wvol[w] += np.bincount(sid[msk], weights=v[msk], minlength=ns)
            wbv[w]  += np.bincount(sid[msk], weights=(v[msk]*buy[msk]), minlength=ns)
            wsv[w]  += np.bincount(sid[msk], weights=(v[msk]*(~buy[msk])), minlength=ns)
    print('  tx passA', round(time.time()-t0), flush=True)
    # pass B: prices at min/max sbp and nearest 15/30
    dH = {h: np.full(ns, np.inf, np.float32) for h in (15,30)}
    t0=time.time()
    for sid, sbp in ziter(path, ['sample_id','seconds_before_predict']):
        for h in dH:
            np.minimum.at(dH[h], sid, np.abs(sbp - h))
    lastp = np.zeros(ns, np.float32); firstp = np.zeros(ns, np.float32)
    pH = {h: np.zeros(ns, np.float32) for h in dH}
    for sid, sbp, price in ziter(path, ['sample_id','seconds_before_predict','price']):
        m = (sbp == mn[sid]); lastp[sid[m]] = price[m]
        m = (sbp == mx[sid]); firstp[sid[m]] = price[m]
        for h in dH:
            m = (np.abs(sbp - h) == dH[h][sid]); pH[h][sid[m]] = price[m]
    print('  tx passB', round(time.time()-t0), flush=True)
    vwap = pv/np.maximum(vol,1)
    F['tx_n']=n; F['tx_vol']=vol; F['tx_imb']=(bvol-svol)/(bvol+svol+1)
    F['tx_vwap']=vwap; F['tx_lastp']=lastp; F['tx_firstp']=firstp
    F['tx_ret']=np.where(firstp>0, lastp/np.maximum(firstp,1e-9)-1, 0)
    F['tx_vwap_dev']=np.where(lastp>0, vwap/np.maximum(lastp,1e-9)-1, 0)
    F['tx_avg_trade']=vol/np.maximum(n,1)
    F['tx_max_trade']=maxv
    F['tx_px_std']=np.sqrt(np.maximum(p2v/np.maximum(vol,1)-vwap*vwap,0))/np.maximum(vwap,1e-9)
    F['tx_recency']=sbpv/np.maximum(vol,1)
    for h in (15,30):
        F[f'tx_ret{h}']=np.where(pH[h]>0, lastp/np.maximum(pH[h],1e-9)-1, 0)
    for w in W:
        F[f'tx_vol{w}_frac']=wvol[w]/np.maximum(vol,1)
        F[f'tx_imb{w}']=(wbv[w]-wsv[w])/(wbv[w]+wsv[w]+1)
    return F

def agg_order(path, ns):
    F = {}
    nb=np.zeros(ns); ns_=np.zeros(ns); cb=np.zeros(ns); cs=np.zeros(ns)
    nnew=np.zeros(ns); ncan=np.zeros(ns)
    pvb=np.zeros(ns); pvs=np.zeros(ns)
    nb30=np.zeros(ns); ns30=np.zeros(ns); cb30=np.zeros(ns); cs30=np.zeros(ns)
    t0=time.time()
    for sid, price, v, side, act in ziter(path, ['sample_id','price','volume','side','order_action']):
        new = act==0; buy = side==0
        nb += np.bincount(sid, weights=v*(new&buy), minlength=ns)
        ns_+= np.bincount(sid, weights=v*(new&~buy), minlength=ns)
        cb += np.bincount(sid, weights=v*(~new&buy), minlength=ns)
        cs += np.bincount(sid, weights=v*(~new&~buy), minlength=ns)
        nnew+=np.bincount(sid, weights=new.astype(np.float64), minlength=ns)
        ncan+=np.bincount(sid, weights=(~new).astype(np.float64), minlength=ns)
        pvb+= np.bincount(sid, weights=price.astype(np.float64)*v*(new&buy), minlength=ns)
        pvs+= np.bincount(sid, weights=price.astype(np.float64)*v*(new&~buy), minlength=ns)
        m30 = sid[sbp30] if False else None
    # second quick pass for 30s window needs sbp; redo with sbp included
    print('  ord passA', round(time.time()-t0), flush=True)
    for sid, v, side, act, sbp in ziter(path, ['sample_id','volume','side','order_action','seconds_before_predict']):
        new = act==0; buy = side==0; m = sbp<=30
        s = sid[m]
        nb30+= np.bincount(s, weights=(v*(new&buy))[m], minlength=ns)
        ns30+= np.bincount(s, weights=(v*(new&~buy))[m], minlength=ns)
        cb30+= np.bincount(s, weights=(v*(~new&buy))[m], minlength=ns)
        cs30+= np.bincount(s, weights=(v*(~new&~buy))[m], minlength=ns)
    print('  ord passB', round(time.time()-t0), flush=True)
    F['ord_new_imb']=(nb-ns_)/(nb+ns_+1)
    F['ord_net_imb']=((nb-cb)-(ns_-cs))/(nb+ns_+cb+cs+1)
    F['ord_nnew']=nnew; F['ord_ncan']=ncan
    F['ord_cancel_ratio']=ncan/np.maximum(nnew+ncan,1)
    F['ord_buy_px']=pvb/np.maximum(nb,1); F['ord_sell_px']=pvs/np.maximum(ns_,1)
    F['ord_px_spread']=(F['ord_sell_px']-F['ord_buy_px'])/np.maximum((F['ord_sell_px']+F['ord_buy_px'])/2,1e-9)
    F['ord_new_imb30']=(nb30-ns30)/(nb30+ns30+1)
    F['ord_net_imb30']=((nb30-cb30)-(ns30-cs30))/(nb30+ns30+cb30+cs30+1)
    F['ord_vol30_frac']=(nb30+ns30+cb30+cs30)/np.maximum(nb+ns_+cb+cs,1)
    return F

def agg_market(path, ns):
    cols = ['sample_id','seconds_before_predict','transaction_avgprice','transaction_volume','transaction_count',
            'ask_price_1','bid_price_1','ask_volume_1','bid_volume_1','ask_volume_2','bid_volume_2']
    mn  = np.full(ns, np.inf, np.float32); mx = np.full(ns, -np.inf, np.float32)
    dH = {h: np.full(ns, np.inf, np.float32) for h in (30,60,120,300)}
    nbars = np.zeros(ns)
    cumvol_max = np.zeros(ns); cumcnt_max = np.zeros(ns)
    st=np.zeros(ns); st2=np.zeros(ns); sx=np.zeros(ns); sx2=np.zeros(ns); stx=np.zeros(ns)
    sm=np.zeros(ns); sm2=np.zeros(ns); stm=np.zeros(ns)
    sspread=np.zeros(ns); srelspread=np.zeros(ns); simb1=np.zeros(ns); simb2=np.zeros(ns)
    t0=time.time()
    for ch in ziter(path, cols):
        sid, sbp, ap, tv, tc, a1, b1, av1, bv1, a2, b2, av2, bv2 = ch
        nbars += np.bincount(sid, minlength=ns)
        np.minimum.at(mn, sid, sbp); np.maximum.at(mx, sid, sbp)
        for h in dH:
            np.minimum.at(dH[h], sid, np.abs(sbp - h))
        np.maximum.at(cumvol_max, sid, tv)
        np.maximum.at(cumcnt_max, sid, tc)
        t = sbp.astype(np.float64); x = ap.astype(np.float64)
        mid = (a1.astype(np.float64)+b1)/2
        st+=np.bincount(sid,weights=t,minlength=ns); st2+=np.bincount(sid,weights=t*t,minlength=ns)
        sx+=np.bincount(sid,weights=x,minlength=ns); sx2+=np.bincount(sid,weights=x*x,minlength=ns)
        stx+=np.bincount(sid,weights=t*x,minlength=ns)
        sm+=np.bincount(sid,weights=mid,minlength=ns); sm2+=np.bincount(sid,weights=mid*mid,minlength=ns)
        stm+=np.bincount(sid,weights=t*mid,minlength=ns)
        sspread+=np.bincount(sid,weights=(a1-b1).astype(np.float64),minlength=ns)
        srelspread+=np.bincount(sid,weights=(a1-b1)/np.maximum(mid,1e-9),minlength=ns)
        imb1=(bv1-av1)/(bv1+av1+1); bv=bv1+bv2; av=av1+av2; imb2=(bv-av)/(bv+av+1)
        simb1+=np.bincount(sid,weights=imb1,minlength=ns)
        simb2+=np.bincount(sid,weights=imb2,minlength=ns)
    print('  mk passA', round(time.time()-t0), flush=True)
    # pass B: values at min sbp (last bar), max sbp (first bar), horizons
    atmin = {k: np.zeros(ns) for k in ['ap','tv','tc','a1','b1','av1','bv1','av2','bv2']}
    atmax = {k: np.zeros(ns) for k in ['ap','tv']}
    atH = {h: {k: np.zeros(ns) for k in ['ap','tv','a1','b1']} for h in dH}
    t0=time.time()
    for ch in ziter(path, cols):
        sid, sbp, ap, tv, tc, a1, b1, av1, bv1, a2, b2, av2, bv2 = ch
        vals = dict(zip(['ap','tv','tc','a1','b1','av1','bv1','av2','bv2'], ch[2:]))
        m = (sbp == mn[sid]); s = sid[m]
        for k in atmin: atmin[k][s] = vals[k][m]
        m = (sbp == mx[sid]); s = sid[m]
        for k in atmax: atmax[k][s] = vals[k][m]
        for h in dH:
            m = (np.abs(sbp - h) == dH[h][sid]); s = sid[m]
            for k in atH[h]: atH[h][k][s] = vals[k][m]
    print('  mk passB', round(time.time()-t0), flush=True)
    F = {}
    mid = (atmin['a1']+atmin['b1'])/2
    F['mk_spread']=(atmin['a1']-atmin['b1'])/np.maximum(mid,1e-9)
    F['mk_imb1']=(atmin['bv1']-atmin['av1'])/(atmin['bv1']+atmin['av1']+1)
    bv=atmin['bv1']+atmin['bv2']; av=atmin['av1']+atmin['av2']
    F['mk_imb2']=(bv-av)/(bv+av+1)
    F['mk_ret10m']=np.where(atmax['ap']>0, atmin['ap']/np.maximum(atmax['ap'],1e-9)-1, 0)
    F['mk_vol']=atmin['tv']; F['mk_cnt']=atmin['tc']
    F['mk_avgpx_dev']=np.where(mid>0, atmin['ap']/np.maximum(mid,1e-9)-1, 0)
    # horizon returns (avgprice and mid) + volume windows
    midl = (atmin['a1']+atmin['b1'])/2
    for h in (30,60,120,300):
        aph = atH[h]['ap']; midh=(atH[h]['a1']+atH[h]['b1'])/2
        F[f'mk_ret{h}']=np.where(aph>0, atmin['ap']/np.maximum(aph,1e-9)-1, 0)
        F[f'mk_midret{h}']=np.where(midh>0, midl/np.maximum(midh,1e-9)-1, 0)
    F['mk_vol_last60']=atmin['tv']-atH[60]['tv']
    F['mk_vol_60_300']=atH[60]['tv']-atH[300]['tv']
    F['mk_vol_accel']=(F['mk_vol_last60']/60)/(np.maximum(F['mk_vol_60_300'],0)/240+1)
    F['mk_vol_total']=cumvol_max; F['mk_cnt_total']=cumcnt_max
    F['mk_nbars']=nbars
    nn = np.maximum(nbars,1)
    mt = st/nn; mt2 = st2/nn
    var_t = np.maximum(mt2-mt*mt,1e-9)
    mxp = sx/nn
    F['mk_slope_px']=(stx/nn - mt*mxp)/var_t          # d(avgprice)/d(sbp); negative => rising into prediction
    F['mk_px_vol']=np.sqrt(np.maximum(sx2/nn-mxp*mxp,0))/np.maximum(mxp,1e-9)
    mm = sm/nn
    F['mk_slope_mid']=(stm/nn - mt*mm)/var_t
    F['mk_mid_vol']=np.sqrt(np.maximum(sm2/nn-mm*mm,0))/np.maximum(mm,1e-9)
    F['mk_spread_mean']=sspread/nn/np.maximum(mm,1e-9)
    F['mk_relspread_mean']=srelspread/nn
    F['mk_imb1_mean']=simb1/nn
    F['mk_imb2_mean']=simb2/nn
    return F

if __name__ == '__main__':
    split, ns = sys.argv[1], int(sys.argv[2])
    base = f'/tmp/mscapital/{split}'
    out = {}
    for name, fn, file in [('tx', agg_transaction, 'transaction.feather'),
                           ('ord', agg_order, 'order.feather'),
                           ('mk', agg_market, 'market.feather')]:
        t=time.time()
        F = fn(f'{base}/{file}', ns)
        out.update(F)
        print(f'{split} {name} done {time.time()-t:.0f}s total_feats={len(out)}', flush=True)
        gc.collect()
    keys = sorted(out)
    X = np.stack([out[k] for k in keys], axis=1).astype(np.float32)
    np.save(f'/tmp/work/X2_{split}.npy', X)
    np.save(f'/tmp/work/X2_{split}_keys.npy', np.array(keys))
    print('saved', X.shape, flush=True)
    print(keys, flush=True)