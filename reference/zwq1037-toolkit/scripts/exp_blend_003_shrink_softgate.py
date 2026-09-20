"""Forward shrink stacking and a single soft volatility gate on saved predictions."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE'
NAMES=['tabm','realmlp','transformer','gru']
def cosine(y,p):
    return float(y@p/(np.linalg.norm(y)*np.linalg.norm(p)+1e-30))
def fit_weights(x,y):
    # 固定正则强度，不用后半段挑参数。
    # Fix regularization without selecting parameters on the forward block.
    penalty=np.sqrt(0.1*np.trace(x.T@x)/x.shape[1])
    w,_=nnls(np.vstack([x,penalty*np.eye(x.shape[1])]),np.r_[y,np.zeros(x.shape[1])])
    return w/w.sum()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tabm=pd.read_feather(ROOT/'data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather')
    tabm=tabm.loc[tabm.month.ne(66),['sample_id','month','target','candidate']].rename(columns={'candidate':'tabm'})
    rd=ROOT/'data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66'
    real=pd.DataFrame({'sample_id':np.load(rd/'validation_sample_ids.npy'),'realmlp':np.load(rd/'validation_predictions.npy')})
    tr=pd.read_csv(ROOT/'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv',usecols=['sample_id','prediction']).rename(columns={'prediction':'transformer'})
    gr=pd.read_csv(ROOT/'data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv',usecols=['sample_id','prediction']).rename(columns={'prediction':'gru'})
    frame=tabm
    for source in (real,tr,gr):
        frame=frame.merge(source,on='sample_id',how='left',validate='one_to_one')
    assert len(frame)==140806 and frame[NAMES].notna().all().all()
    x=frame[NAMES].to_numpy(np.float64); y=frame.target.to_numpy(np.float64); months=frame.month.to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    select=np.isin(months,[62,63,64,65]); forward=np.isin(months,[67,68,69,70])
    scale=np.sqrt(np.mean(x[select]**2,axis=0)); z=x/scale
    target_scale=np.sqrt(np.mean(y[select]**2)); ys=y/target_scale
    meta=json.loads((ROOT/'outputs/submission_metadata/current151_theoretical_geometry_optimum.json').read_text(encoding='utf-8'))
    old=meta['best_weights']
    base=np.array([old['tabm'],old['realmlp'],old['transformer_old'],old['gru']]);base/=base.sum()
    learned=fit_weights(z[select],ys[select]); shrink=.75*base+.25*learned
    predictions={'fixed_current_owned':z@base,'unshrunk_nnls':z@learned,'shrink25':z@shrink}
    metrics={}
    for name,p in predictions.items():
        metrics[name]={'selection':cosine(y[select],p[select]),'forward':cosine(y[forward],p[forward]),'all':cosine(y,p),'monthly_forward':{str(m):cosine(y[months==m],p[months==m]) for m in range(67,71)}}
    delta=metrics['shrink25']['forward']-metrics['fixed_current_owned']['forward']
    monthly_delta={str(m):metrics['shrink25']['monthly_forward'][str(m)]-metrics['fixed_current_owned']['monthly_forward'][str(m)] for m in range(67,71)}
    stack_pass=delta>=.0003 and sum(v>=0 for v in monthly_delta.values())>=3
    stack={'fixed_weights':dict(zip(NAMES,base.tolist())),'learned_weights':dict(zip(NAMES,learned.tolist())),'shrink_weights':dict(zip(NAMES,shrink.tolist())),'metrics':metrics,'forward_delta':delta,'monthly_delta':monthly_delta,'passed':stack_pass}
    print('STACK '+json.dumps(stack),flush=True)
    # 一个固定的软门控候选：沿用 rv60 三分位状态，不新增行情变量。
    # One fixed soft-gating candidate with the existing rv60 tertile states.
    rv=pd.read_feather(ROOT/'data/processed/train_market_microstructure_features.feather',columns=['sample_id','realized_volatility_60'])
    frame=frame.merge(rv,on='sample_id',how='left',validate='one_to_one')
    v=frame.realized_volatility_60.to_numpy(np.float64)
    median=float(np.nanmedian(v[select]));v=np.nan_to_num(v,nan=median,posinf=median,neginf=median)
    thresholds=np.quantile(v[select],[1/3,2/3]);bins=np.digitize(v,thresholds,right=True)
    anchors=np.array([np.median(v[select&(bins==b)]) for b in range(3)])
    state_weights=np.vstack([.75*base+.25*fit_weights(z[select&(bins==b)],ys[select&(bins==b)]) for b in range(3)])
    assert np.isfinite(anchors).all() and np.all(np.diff(anchors)>0)
    assert np.isfinite(state_weights).all()
    sample_weights=np.column_stack([np.interp(v,anchors,state_weights[:,j]) for j in range(4)])
    assert np.isfinite(sample_weights).all() and np.allclose(sample_weights.sum(axis=1),1)
    soft=(z*sample_weights).sum(axis=1);hard=(z*state_weights[bins]).sum(axis=1)
    fixed=predictions['fixed_current_owned']
    soft_delta=cosine(y[forward],soft[forward])-cosine(y[forward],fixed[forward])
    soft_month={str(m):cosine(y[months==m],soft[months==m])-cosine(y[months==m],fixed[months==m]) for m in range(67,71)}
    soft_pass=soft_delta>=.0003 and sum(d>=0 for d in soft_month.values())>=3
    gate={'anchors':anchors.tolist(),'thresholds':thresholds.tolist(),'state_weights':state_weights.tolist(),'forward_fixed':cosine(y[forward],fixed[forward]),'forward_hard_control':cosine(y[forward],hard[forward]),'forward_soft':cosine(y[forward],soft[forward]),'forward_delta':soft_delta,'monthly_delta':soft_month,'passed':soft_pass}
    for name,p in predictions.items():frame[name]=p
    frame['soft_gate']=soft;frame['hard_gate']=hard
    frame.to_feather(OUT/'validation_predictions.feather')
    summary={'experiment':'EXP-BLEND-003-SHRINK-SOFTGATE','status':'complete','models':NAMES,'selection_months':'62-65','forward_months':'67-70','ridge_ratio':.1,'shrink_fraction':.25,'prediction_scale':scale.tolist(),'stacking':stack,'soft_gate':gate,'limitation':'Only owned models have aligned validation predictions; public block is not evaluated.'}
    (OUT/'score_only.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez(OUT/'fusion_parameters.npz',scale=scale,base=base,shrink=shrink,anchors=anchors,state_weights=state_weights,rv_median=median)
    print('SOFT_GATE '+json.dumps(gate),flush=True)
if __name__=='__main__':main()
