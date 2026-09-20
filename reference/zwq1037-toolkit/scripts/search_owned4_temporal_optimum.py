"""Optimize four owned models on saved temporal deployment predictions."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/interim/tree_experiments/EXP-BLEND-OWNED-20260917'
OUT=ROOT/'data/interim/tree_experiments/EXP-BLEND-004-OWNED4-OPTIMUM'
NAMES=['tabm','gru','realmlp','transformer']
def compositions(total):
    for a in range(total+1):
        for b in range(total-a+1):
            for c in range(total-a-b+1):yield (a,b,c,total-a-b-c)
def terms(x,y,mask):
    a=x[mask];b=y[mask];return a.T@a,a.T@b,b@b
def scores(w,t):
    gram,cross,yy=t
    return (w@cross)/np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',w,gram,w)*yy,1e-30))
def fit(x,y,ridge=0):
    penalty=np.sqrt(ridge*np.trace(x.T@x)/4)
    w,_=nnls(np.vstack([x,penalty*np.eye(4)]),np.r_[y,np.zeros(4)])
    assert w.sum()>0
    return w/w.sum()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    f=pd.read_feather(BASE/'recovered_temporal/owned_temporal3fold_valid6570.feather')
    stats=json.loads((BASE/'temporal3fold/score_only.json').read_text(encoding='utf-8'))
    rms=np.array([stats['selection_rms'][n] for n in NAMES])
    x=f[NAMES].to_numpy(np.float64)/rms;y=f.target.to_numpy(np.float64);months=f.month.to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    selection=months<=67;forward=months>=68
    masks={'selection65_67':selection,'forward68_70':forward,'all65_70':np.ones(len(f),bool)}
    masks.update({f'month{m}':months==m for m in np.unique(months)})
    ts={name:terms(x,y,mask) for name,mask in masks.items()}
    prior=np.array([.33,.20,.28,.19])
    grid=np.array(list(compositions(100)),dtype=np.float64)/100
    grid_scores=scores(grid,ts['selection65_67'])
    best_grid=grid[np.argmax(grid_scores)]
    fit_selection=fit(x[selection],y[selection]);fit_all=fit(x,y)
    candidates={'prior_scored_0147':prior,'equal':np.ones(4)/4,'grid_selection_best':best_grid,'continuous_selection_best':fit_selection,'exploratory_all_local_optimum':fit_all}
    for ridge in (.01,.1,1.):
        learned=fit(x[selection],y[selection],ridge)
        candidates['ridge'+str(ridge)]=learned
        for fraction in (.25,.5,.75):candidates[f'ridge{ridge}_shrink{fraction}']=(1-fraction)*prior+fraction*learned
    reports={name:{'weights':dict(zip(NAMES,w.tolist())),'scores':{key:float(scores(w[None],t)[0]) for key,t in ts.items()}} for name,w in candidates.items()}
    # 理论本地最优与前向最优分别标记，不把已查看数据当新 holdout。
    # Label retrospective and forward results separately; these months are previously inspected.
    selection_name=max((name for name in candidates if name!='exploratory_all_local_optimum'),key=lambda name:reports[name]['scores']['selection65_67'])
    forward_name=max((name for name in candidates if name!='exploratory_all_local_optimum'),key=lambda name:reports[name]['scores']['forward68_70'])
    table=pd.DataFrame(grid,columns=NAMES)
    for key,t in ts.items():table[key]=scores(grid,t)
    table.sort_values('selection65_67',ascending=False).to_csv(OUT/'owned4_grid_1percent.csv',index=False)
    incumbent_meta=json.loads((ROOT/'outputs/submission_metadata/owned_only_tabm33_gru20_realmlp28_transformer19.json').read_text(encoding='utf-8'))
    test={};ids=None
    def read(path):
        nonlocal ids
        part=pd.read_csv(path);current=part.sample_id.to_numpy()
        if ids is None:ids=current
        assert len(part)==647896 and np.array_equal(ids,current)
        values=part.prediction.to_numpy(float);assert np.isfinite(values).all()
        return values
    for name in ['tabm','realmlp','transformer']:test[name]=read(incumbent_meta['sources'][name])/rms[NAMES.index(name)]
    parts=[]
    for fold in ('52','57','62'):
        norm=incumbent_meta['sources']['gru'][fold]
        parts.append((read(norm['source'])-norm['mean'])/norm['std'])
    test['gru']=np.mean(parts,axis=0)/rms[1]
    test_x=np.column_stack([test[n] for n in NAMES])
    # 检查现有0.147配方重建一致，确保测试缩放没改变。
    # Reconstruct the scored 0.147 recipe to check frozen deployment scaling.
    old=pd.read_csv(ROOT/'outputs/submissions/owned_only_tabm33_gru20_realmlp28_transformer19.csv')
    assert np.array_equal(ids,old.sample_id.to_numpy()) and np.allclose(test_x@prior,old.prediction.to_numpy(),atol=1e-10)
    exported={}
    for label,name in [('selection',selection_name),('forward_diagnostic',forward_name),('all_local_optimum','exploratory_all_local_optimum')]:
        path=ROOT/'outputs/submissions'/f'owned4_{label}_20260917.csv'
        prediction=test_x@candidates[name]
        assert np.isfinite(prediction).all()
        pd.DataFrame({'sample_id':ids,'prediction':prediction}).to_csv(path,index=False)
        exported[label]={'recipe':name,'path':str(path),'correlation_with_prior':float(np.corrcoef(prediction,old.prediction.to_numpy())[0,1])}
    report={'status':'complete','models':NAMES,'public_weight':0,'tree_weight':0,'rows':len(f),'grid_count':len(grid),'selection_months':[65,67],'forward_months':[68,69,70],'normalization':'Frozen prior temporal validation RMS and per-fold GRU mean/std, identical to scored pure-owned0.147 CSV','recipes':reports,'selection_winner':selection_name,'best_forward_among_predeclared':forward_name,'exports':exported,'limitation':'All-local optimum is fitted and scored on the same data. Forward-diagnostic selection also examines already inspected months. Neither proves leaderboard optimality. RealMLP and Transformer use development validation counterparts with fulltrain test predictions.'}
    (OUT/'score_only.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'outputs/submission_metadata/owned4_weight_search_20260917.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'grid_count':len(grid),'selection_winner':selection_name,'forward_winner':forward_name,'main_recipes':{k:reports[k] for k in {'prior_scored_0147',selection_name,forward_name,'exploratory_all_local_optimum'}},'exports':exported},indent=2),flush=True)
if __name__=='__main__':main()
