"""Apply the fixed alpha-100 month common component to the current 0.138 recipe."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from analyze_conv_hybrid_transformer_full_blend import load_window
from exp_tabm_014_quantile_stage_curves import unit
PROJECT=Path(__file__).resolve().parents[1]
REF=PROJECT/'data'/'interim'/'kaggle_kernels'/'relative319_xs_tabm_notebook'; sys.path.insert(0,str(REF))
import run_extracted as recipe
ALPHA=100.0

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def table(base,months,target,ix):
 rows=[]; ys=[]; labels=[]
 for m in np.unique(months[ix]):
  j=ix[months[ix]==m]; v=np.asarray(base[j],dtype=np.float64)
  rows.append(np.r_[np.nanmean(v,axis=0),np.nanstd(v,axis=0)]); ys.append(target[j].mean()); labels.append(int(m))
 return np.asarray(rows),np.asarray(ys),np.asarray(labels)
def common_predictions(base,months,target,train_end,valid_months):
 top=[columns.index(c) for c in recipe.TOP_FEATURES]
 tr=np.flatnonzero(months<=train_end); va=np.flatnonzero(np.isin(months,valid_months))
 a,y,_=table(base[:,top],months,target,tr); b,_,bm=table(base[:,top],months,target,va)
 med=np.nanmedian(a,axis=0); a=np.where(np.isfinite(a),a,med); b=np.where(np.isfinite(b),b,med)
 sc=StandardScaler().fit(a); model=Ridge(alpha=ALPHA).fit(sc.transform(a),y)
 return dict(zip(bm,model.predict(sc.transform(b))))
def correct(pred,months,means):
 out=pred.copy()
 for m in np.unique(months):
  z=months==m; out[z]=out[z]-out[z].mean()+means[int(m)]
 return out
def main():
 global columns
 labels=pd.read_feather(PROJECT/'data'/'raw'/'label.feather',columns=['sample_id','month','target']).sort_values('sample_id').reset_index(drop=True)
 all_months=labels.month.to_numpy(); all_target=labels.target.to_numpy(dtype=np.float64)
 base=np.load(PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'features.npy',mmap_mode='r')
 columns=json.loads((PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'feature_columns.json').read_text())
 rows=[]
 for name,fold,start,end,train_end in [('first','train049_valid5059',50,59,49),('second','train059_valid6070',60,70,59)]:
  current,parts,target,months,_=load_window(fold,start,end)
  trans=.335*parts['hybrid42']+.335*parts['hybrid137']+.33*parts['conv42']
  current138=.70*unit(current)+.30*unit(trans)
  xs=pd.read_csv(REF/'output_v1'/f'{fold}_predictions.csv').relative319_xs40.to_numpy(dtype=np.float64)
  vm=np.arange(start,end+1); means=common_predictions(base,all_months,all_target,train_end,vm)
  mask=np.ones(len(months),dtype=bool) if name=='first' else (months>=62)&(months!=66)
  for w in (0.0,.10,.20,.30,.40):
   pred=(1-w)*unit(current138)+w*unit(xs)
   adjusted=correct(pred,months,means)
   rows.append({'window':name,'xs_weight':w,'raw':cos(target[mask],pred[mask]),'common_corrected':cos(target[mask],adjusted[mask]),'common_delta':cos(target[mask],adjusted[mask])-cos(target[mask],pred[mask])})
 out=pd.DataFrame(rows); out.to_csv(PROJECT/'outputs'/'common_component_current138.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
