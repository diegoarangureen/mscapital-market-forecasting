"""X33 step 2: X32 report recomputed with the champion recipe (direct mean of seed predictions, not unit-norm). X32 stays closed."""
import numpy as np, json
y=np.load('/tmp/prepared/y.npy').astype(np.float64); month=np.load('/tmp/prepared/month.npy')
V={2026:'/tmp/x32/v2/jobs/',42:'/tmp/x32/v3/jobs/'}
cos=lambda p,t:float(p@t/np.sqrt((p@p)*(t@t)))
def load(arm,f,s):
    z=np.load(f'{V[s]}{arm}_o70_f{f}_s{s}/predictions.npz'); return z['score_rows'],z['score_pred'].astype(np.float64)
out={}
for f in (0,4):
    r={}
    for arm in ('flow31','flow31ctx'):
        rows,p26=load(arm,f,2026); rows2,p42=load(arm,f,42); assert (rows==rows2).all()
        r[arm]=dict(rows=rows,s26=p26,s42=p42,mean=(p26+p42)/2,unit=p26/np.linalg.norm(p26)+p42/np.linalg.norm(p42),
                    n26=float(np.linalg.norm(p26)),n42=float(np.linalg.norm(p42)))
    rows=r['flow31']['rows']; assert (rows==r['flow31ctx']['rows']).all(); yy=y[rows]; mm=month[rows]
    d={'n':len(rows)}
    for k in ('s26','s42','mean','unit'):
        a,b=cos(r['flow31'][k],yy),cos(r['flow31ctx'][k],yy); d[k]=dict(flow31=a,ctx=b,delta=b-a)
    d['norm_ratio_s26_s42_flow31']=r['flow31']['n26']/r['flow31']['n42']; d['norm_ratio_s26_s42_ctx']=r['flow31ctx']['n26']/r['flow31ctx']['n42']
    d['months_direct_delta']={int(m):cos(r['flow31ctx']['mean'][mm==m],yy[mm==m])-cos(r['flow31']['mean'][mm==m],yy[mm==m]) for m in np.unique(mm)}
    out[f]=d
json.dump(out,open('/tmp/repo/research/x33/x32_direct_mean.json','w'),indent=1)
for f,d in out.items():
    print('fold',f,'n',d['n'])
    for k in ('s26','s42','mean','unit'): print(' ',k,{a:round(b,6) for a,b in d[k].items()})
    print('  norm ratios',round(d['norm_ratio_s26_s42_flow31'],3),round(d['norm_ratio_s26_s42_ctx'],3))
    print('  months',{m:round(v,5) for m,v in d['months_direct_delta'].items()})
