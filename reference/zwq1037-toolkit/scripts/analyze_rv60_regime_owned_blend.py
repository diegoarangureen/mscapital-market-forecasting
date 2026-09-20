from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path('.')
def unit(x):
    x=np.asarray(x,np.float64)
    return x/np.linalg.norm(x)
def cosine(y,p):
    d=np.linalg.norm(y)*np.linalg.norm(p)
    return float(y@p/d) if d else 0.0

tabm=pd.read_feather(P/'data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather')
tabm=tabm.loc[tabm.month.ne(66),['sample_id','month','target','candidate']].rename(columns={'candidate':'tabm'})
rd=P/'data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66'
real=pd.DataFrame({'sample_id':np.load(rd/'validation_sample_ids.npy'),'realmlp':np.load(rd/'validation_predictions.npy')})
tr=pd.read_csv(P/'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv',usecols=['sample_id','prediction']).rename(columns={'prediction':'transformer'})
gr=pd.read_csv(P/'data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv',usecols=['sample_id','prediction']).rename(columns={'prediction':'gru'})
rv=pd.read_feather(P/'data/processed/train_market_microstructure_features.feather',columns=['sample_id','realized_volatility_60'])
f=tabm.merge(real,on='sample_id').merge(tr,on='sample_id').merge(gr,on='sample_id').merge(rv,on='sample_id')
names=['tabm','realmlp','transformer','gru']
z=np.column_stack([unit(f[n]) for n in names])
y=f.target.to_numpy(np.float64); months=f.month.to_numpy(); v=f.realized_volatility_60.to_numpy(np.float64)
base=np.array([.12,.03,.18,.03]); base=base/base.sum()
recipes={
'base':base,
'more_transformer':np.array([.25,.05,.65,.05]),
'more_tabm':np.array([.50,.05,.40,.05]),
'more_gru':np.array([.30,.05,.45,.20]),
'more_realmlp':np.array([.30,.20,.45,.05]),
'balanced':np.array([.30,.15,.45,.10]),
'tabm_transformer':np.array([.40,0,.60,0]),
}
select=(months>=62)&(months<=65)
forward=(months>=67)&(months<=70)
q=np.quantile(v[select & np.isfinite(v)],[1/3,2/3])
bins=np.digitize(np.nan_to_num(v,nan=np.nanmedian(v[select])),q)
chosen={}
dynamic=np.empty(len(f),np.float64)
for b in range(3):
    m=select&(bins==b)
    scores={k:cosine(y[m],z[m]@w) for k,w in recipes.items()}
    best=max(scores,key=scores.get)
    chosen[str(b)]={'recipe':best,'selection_cosine':scores[best],'base_cosine':scores['base'],'selection_delta':scores[best]-scores['base'],'scores':scores}
    dynamic[bins==b]=z[bins==b]@recipes[best]
base_pred=z@base
fixed_scores={k:{'selection':cosine(y[select],z[select]@w),'forward':cosine(y[forward],z[forward]@w)} for k,w in recipes.items()}
report={
 'regime':'tertiles of realized_volatility_60 fitted on months62-65',
 'thresholds':q.tolist(),
 'chosen':chosen,
 'overall':{
  'selection_base':cosine(y[select],base_pred[select]),
  'selection_dynamic':cosine(y[select],dynamic[select]),
  'selection_delta':cosine(y[select],dynamic[select])-cosine(y[select],base_pred[select]),
  'forward_base':cosine(y[forward],base_pred[forward]),
  'forward_dynamic':cosine(y[forward],dynamic[forward]),
  'forward_delta':cosine(y[forward],dynamic[forward])-cosine(y[forward],base_pred[forward]),
 },
 'monthly_forward_delta':{str(m):cosine(y[months==m],dynamic[months==m])-cosine(y[months==m],base_pred[months==m]) for m in range(67,71)},
 'fixed_recipe_scores':fixed_scores,
 'limitation':'owned 36% proxy only; public block has no OOF'
}
out=P/'outputs/submission_metadata/rv60_regime_owned_blend_forward.json'
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
