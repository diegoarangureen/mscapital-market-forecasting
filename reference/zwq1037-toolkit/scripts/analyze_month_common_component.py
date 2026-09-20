"""Test a deployable month-level common component from input-only group summaries."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
PROJECT=Path(__file__).resolve().parents[1]
REF=PROJECT/'data'/'interim'/'kaggle_kernels'/'relative319_xs_tabm_notebook'
sys.path.insert(0,str(REF))
import run_extracted as recipe


def cos(a,b):
 a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
 return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def month_table(base,months,target,indices):
 rows=[]; ys=[]; labels=[]
 for m in np.unique(months[indices]):
  ix=indices[months[indices]==m]; v=np.asarray(base[ix],dtype=np.float64)
  rows.append(np.concatenate([np.nanmean(v,axis=0),np.nanstd(v,axis=0)])); ys.append(float(np.mean(target[ix]))); labels.append(int(m))
 return np.asarray(rows),np.asarray(ys),np.asarray(labels)

def run_fold(base,months,target,pred,train_end,vstart,vend,exclude66):
 train=np.flatnonzero(months<=train_end); valid=np.flatnonzero((months>=vstart)&(months<=vend))
 top=[columns.index(c) for c in recipe.TOP_FEATURES]
 xtr,ytr,mtr=month_table(base[:,top],months,target,train)
 xva,yva,mva=month_table(base[:,top],months,target,valid)
 med=np.nanmedian(xtr,axis=0); xtr=np.where(np.isfinite(xtr),xtr,med); xva=np.where(np.isfinite(xva),xva,med)
 scaler=StandardScaler().fit(xtr); xtr=scaler.transform(xtr); xva=scaler.transform(xva)
 report={}
 score_rows=valid if not exclude66 else valid[months[valid]!=66]
 raw=cos(target[score_rows],pred[score_rows])
 for alpha in (1.0,10.0,100.0,1000.0,10000.0):
  model=Ridge(alpha=alpha).fit(xtr,ytr)
  common=dict(zip(mva,model.predict(xva)))
  adjusted=pred[score_rows].copy()
  for m in np.unique(months[score_rows]):
   mask=months[score_rows]==m
   adjusted[mask]=adjusted[mask]-adjusted[mask].mean()+common[int(m)]
  report[str(alpha)]={'cosine':cos(target[score_rows],adjusted),'delta':cos(target[score_rows],adjusted)-raw,'month_mean_rmse':float(np.sqrt(np.mean((model.predict(xva)-yva)**2)))}
 return {'raw':raw,'alphas':report}

def main():
 global columns
 labels=pd.read_feather(PROJECT/'data'/'raw'/'label.feather',columns=['sample_id','month','target']).sort_values('sample_id').reset_index(drop=True)
 months=labels.month.to_numpy(); target=labels.target.to_numpy(dtype=np.float64)
 base=np.load(PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'features.npy',mmap_mode='r')
 columns=json.loads((PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'feature_columns.json').read_text())
 out={}
 specs=[('first',49,50,59,False,PROJECT/'data'/'interim'/'submissions'/'tabm_relative319_xs40_seed42_holdout049'/'model_holdout049.pt'),('second',59,62,70,True,PROJECT/'data'/'interim'/'submissions'/'tabm_relative319_xs40_seed42_holdout059'/'model_holdout059.pt')]
 # Reuse already saved exact-month validation predictions from prior inference reports by running lightweight model predictions is avoided here.
 for name,te,vs,ve,ex,_ in specs:
  csv=PROJECT/'data'/'interim'/'kaggle_kernels'/'relative319_xs_tabm_notebook'/'output_v1'/('train049_valid5059_predictions.csv' if name=='first' else 'train059_valid6070_predictions.csv')
  frame=pd.read_csv(csv)
  pred=np.zeros(len(target),dtype=np.float64); pred[frame.sample_id.to_numpy(dtype=np.int64)]=frame.relative319_xs40.to_numpy(dtype=np.float64)
  out[name]=run_fold(base,months,target,pred,te,vs,ve,ex)
 path=PROJECT/'data'/'interim'/'month_common_component'; path.mkdir(parents=True,exist_ok=True); (path/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
