# X20: window-delta + cross-stream interaction features derived in-RAM from X2.
# Ideas studied from UnseenAnchor's public repo (feat_microstructure_v3), reimplemented on our stats.
import numpy as np, gc, ctypes
_libc = ctypes.CDLL('libc.so.6')
def build(split):
    X = np.load(f'/tmp/work/X2_{split}.npy')
    k = [str(x) for x in np.load(f'/tmp/work/X2_{split}_keys.npy', allow_pickle=True)]
    i = {name: idx for idx, name in enumerate(k)}
    F = {}; 
    def g(nm): return X[:, i[nm]].astype(np.float64)
    # window deltas (short minus long)
    F['d_tx_imb_15_30']   = g('tx_imb15') - g('tx_imb30')
    F['d_tx_ret_15_30']   = g('tx_ret15') - g('tx_ret30')
    F['d_ret_30_60']      = g('mk_ret30') - g('mk_ret60')
    F['d_ret_60_300']     = g('mk_ret60') - g('mk_ret300')
    F['d_midret_30_60']   = g('mk_midret30') - g('mk_midret60')
    F['d_midret_60_300']  = g('mk_midret60') - g('mk_midret300')
    F['d_volfrac_15_30']  = g('tx_vol15_frac') - g('tx_vol30_frac')
    F['d_ord_new_imb_30'] = g('ord_new_imb30') - g('ord_new_imb')
    F['d_ord_net_imb_30'] = g('ord_net_imb30') - g('ord_net_imb')
    # cross-stream interactions
    F['x_tximb_mimb1']       = g('tx_imb') * g('mk_imb1')
    F['x_tximb30_mimb1mean'] = g('tx_imb30') * g('mk_imb1_mean')
    F['x_ordnew_mimb1']      = g('ord_new_imb') * g('mk_imb1')
    F['x_ordnet_mimb1']      = g('ord_net_imb') * g('mk_imb1')
    F['x_vwapdev_mimb']      = g('tx_vwap_dev') * g('mk_imb1')
    F['x_spread_imb']        = g('mk_spread') * g('mk_imb1')
    F['x_relspread_imbmean'] = g('mk_relspread_mean') * g('mk_imb1_mean')
    F['x_netflow_pressure']  = g('tx_imb') + g('ord_net_imb')
    F['x_t_minus_o']         = g('tx_imb') - g('ord_net_imb')
    F['x_new_minus_cancel']  = g('ord_new_imb') - g('ord_cancel_ratio')
    F['x_midvol_tximb']      = g('mk_mid_vol') * g('tx_imb')
    names = list(F.keys())
    out = np.empty((X.shape[0], len(names)), np.float32)
    for j, nm in enumerate(names):
        v = F[nm]; v[~np.isfinite(v)] = 0.0; out[:, j] = v
    np.save(f'/tmp/work/X20_{split}.npy', out)
    np.save(f'/tmp/work/X20_{split}_names.npy', np.array(names))
    print(split, 'X20 saved', out.shape, flush=True)
if __name__ == '__main__':
    build('train'); gc.collect(); _libc.malloc_trim(0)
    build('test');  gc.collect(); _libc.malloc_trim(0)
    print('X20_DONE', flush=True)
