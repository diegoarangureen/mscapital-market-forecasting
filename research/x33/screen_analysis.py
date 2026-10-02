"""X33 screen analysis: arm vs fresh control, same fold/seed. Direct-mean 2-seed ensemble, cosine on original target."""
import numpy as np, json, os, sys
y=np.load('/tmp/prepared/y.npy').astype(np.float64); month=np.load('/tmp/prepared/month.npy')
cos=lambda p,t:float(p@t/np.sqrt((p@p)*(t@t)))
def load(arm,f,s):
    z=np.load(f'/tmp/x33/{arm}/jobs/flow31_o70_f{f}_s{s}/predictions.npz'); return z['score_rows'],z['score_pred'].astype(np.float64)
arms=[a for a in ('a1','a2') if all(os.path.exists(f'/tmp/x33/{a}/jobs/flow31_o70_f{f}_s{s}/predictions.npz') for f in (0,4) for s in (2026,42))]
out={}
for a in arms:
    res={}; 
    for f in (0,4):
        P={}
        for arm in ('control',a):
            rows,p1=load(arm,f,2026); r2,p2=load(arm,f,42); assert (rows==r2).all(); P[arm]=(rows,p1,p2)
        rows=P['control'][0]; assert (rows==P[a][0]).all(); yy=y[rows]; mm=month[rows]
        d={}
        for i,s in ((1,2026),(2,42)):
            c0,c1=cos(P['control'][i],yy),cos(P[a][i],yy); d[f's{s}']=dict(control=c0,arm=c1,delta=c1-c0)
        e0=(P['control'][1]+P['control'][2])/2; e1=(P[a][1]+P[a][2])/2
        d['ens']=dict(control=cos(e0,yy),arm=cos(e1,yy),delta=cos(e1,yy)-cos(e0,yy))
        d['months']={int(m):cos(e1[mm==m],yy[mm==m])-cos(e0[mm==m],yy[mm==m]) for m in np.unique(mm)}
        d['months_pos']=sum(v>0 for v in d['months'].values()); d['n_months']=len(d['months'])
        res[f]=d
    deltas=[res[f][k]['delta'] for f in (0,4) for k in ('s2026','s42')]
    ens=[res[f]['ens']['delta'] for f in (0,4)]
    allm=[v for f in (0,4) for v in res[f]['months'].values()]
    best_month=max(abs(v) for v in allm)
    res['gate']=dict(each_single_ge_0p002=all(v>=0.002 for v in deltas),ens_pos=all(v>0 for v in ens),
                     mean_single=float(np.mean(deltas)),mean_ens=float(np.mean(ens)),
                     passes=bool(all(v>=0.002 for v in deltas) and all(v>0 for v in ens)))
    out[a]=res
json.dump(out,open('/tmp/repo/research/x33/screen_analysis.json','w'),indent=1)
for a,res in out.items():
    print('==',a)
    for f in (0,4):
        d=res[f]; print(' fold',f,{k:round(d[k]['delta'],5) for k in ('s2026','s42','ens')},'ctrl ens',round(d['ens']['control'],5),'months+',d['months_pos'],'/',d['n_months'])
        print('   months',{m:round(v,4) for m,v in d['months'].items()})
    print(' gate',res['gate'])
