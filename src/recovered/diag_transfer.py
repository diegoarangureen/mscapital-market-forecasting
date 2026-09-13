# val->LB compression diagnosis: label drift, per-month val performance, feature drift (PSI)
import numpy as np, gc
import pyarrow.feather as feather
lab = feather.read_table('/tmp/mscapital/train/label.feather').to_pandas().sort_values('sample_id').reset_index(drop=True)
y = lab.target.values.astype(np.float64); month = lab.month.values
del lab; gc.collect()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

print('== label stats per month (mean/std in bps) ==')
for lo in range(0, 71, 10):
    m = (month>=lo)&(month<=min(lo+9,70))
    print(f'months {lo:2d}-{min(lo+9,70):2d}: n={m.sum():6d} mean={y[m].mean()*1e4:7.3f}bp std={y[m].std()*1e4:7.2f}bp')

print('== per-month val cosine (GBM baseline + MLP + blend) ==')
va = month>=61
yv = y[va]; mv = month[va]
g = np.load('x2base_seedavg_val.npy'); mlp = np.load('mlp_seedavg_val.npy')
gu=(g-g.mean())/g.std(); mu_=(mlp-mlp.mean())/mlp.std(); bl=0.5*gu+0.5*mu_
for mo in range(61,71):
    m = mv==mo
    print(f'month {mo}: gbm {cos(g[m],yv[m]):.4f}  mlp {cos(mlp[m],yv[m]):.4f}  blend {cos(bl[m],yv[m]):.4f}  (n={m.sum()})')

print('== feature drift: PSI train(0-50) vs val(61-70) vs test ==')
X = np.load('X2_train.npy'); keys=list(np.load('X2_train_keys.npy'))
Xte = np.load('X2_test.npy')
tr = month<=50
def psi(a, b):
    qs = np.quantile(a, np.linspace(0,1,11))
    qs[0]-=1e-9; qs[-1]+=1e-9
    ca = np.histogram(a, bins=qs)[0]/len(a); cb = np.histogram(b, bins=qs)[0]/len(b)
    ca=np.maximum(ca,1e-4); cb=np.maximum(cb,1e-4)
    return float(((ca-cb)*np.log(ca/cb)).sum())
rows=[]
for j,k in enumerate(keys):
    rows.append((psi(X[tr,j], X[va,j]), psi(X[tr,j], Xte[:,j]), k))
rows.sort(reverse=True)
print('top drift features (PSI train->val, train->test):')
for pv, pt, k in rows[:12]:
    print(f'  {k:22s} val {pv:.3f}  test {pt:.3f}')
print('DIAG_DONE')
