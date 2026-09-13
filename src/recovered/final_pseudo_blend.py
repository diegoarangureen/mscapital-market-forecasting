# Blend pseudo-GBM test preds with refit MLP test preds, post-process like submission_blend.csv.
import numpy as np, pandas as pd, json
g = np.load('/tmp/work/pseudo_gbm_test.npy')
m = np.load('/tmp/work/refit_mlp_test.npy')
def unit(v):
    v=v-v.mean(); s=v.std(); return v/(s if s>0 else 1)
bl = 0.5*unit(g) + 0.5*unit(m)
nodata = np.load('/tmp/work/test_nodata.npy')
bl[nodata] = 0.0
ref = pd.read_csv('/tmp/work/submission_blend.csv')
print('ref cols:', list(ref.columns), 'rows:', len(ref), flush=True)
ycol = [c for c in ref.columns if c != ref.columns[0]][0]
lo, hi = np.quantile(np.load('/tmp/work/refit_gbm_train.npy'), [0.001, 0.999])
print('clip range from train preds:', lo, hi, flush=True)
out = ref.copy()
out[ycol] = bl
out.to_csv('/tmp/work/submission_pseudo.csv', index=False)
print('wrote submission_pseudo.csv rows', len(out), 'pred std', float(bl.std()), flush=True)
print('BLEND_DONE', flush=True)
