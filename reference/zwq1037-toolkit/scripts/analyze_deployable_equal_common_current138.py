"""Validate exact deployable test-scale month correction with equal sequential groups."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from analyze_conv_hybrid_transformer_full_blend import load_window
PROJECT=Path(__file__).resolve().parents[1]
REF=PROJECT/'data'/'interim'/'kaggle_kernels'/'relative319_xs_tabm_notebook'; sys.path.insert(0,str(REF))
import run_extracted as recipe
ALPHA=100.0
def zunit(v): v=np.asarray(v,float); return (v-v.mean())/v.std()
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def summary(base,groups,target=None):
 rows=[]; ys=[]; keys=[]
 for g in np.unique(groups):
  ix=np.flatnonzero(groups==g); v=np.asarray(base[ix],float)
  rows.append(np.r_[np.nanmean(v,0),np.nanstd(v,0)]); keys.append(int(g))
  if target is not None: ys.append(float(target[ix].mean()))
 return np.asarray(rows),np.asarray(ys),np.asarray(keys)
def apply(pred,groups,common,target_scale):
 out=pred.copy()
 for g in np.unique(groups):
  z=groups==g; out[z]=out[z]-out[z].mean()+common[int(g)]/target_scale
 return out
def main():
 labels=pd.read_feather(PROJECT/'data'/'raw'/'label.feather',columns=['sample_id','month','target']).sort_values('sample_id').reset_index(drop=True)
 allm=labels.month.to_numpy(); ally=labels.target.to_numpy(float); base=np.load(PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'features.npy',mmap_mode='r')
 columns=json.loads((PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'feature_columns.json').read_text()); ix=[columns.index(c) for c in recipe.TOP_FEATURES]; base=base[:,ix]
 rows=[]
 for name,fold,start,end,te in [('first','train049_valid5059',50,59,49),('second','train059_valid6070',60,70,59)]:
  current,parts,target,months,_=load_window(fold,start,end); trans=.335*zunit(parts['hybrid42'])+.335*zunit(parts['hybrid137'])+.33*zunit(parts['conv42']); pred=.7*zunit(current)+.3*zunit(trans)
  tr=allm<=te; va=(allm>=start)&(allm<=end); train_x,train_y,_=summary(base[tr],allm[tr],ally[tr]); med=np.nanmedian(train_x,0); train_x=np.where(np.isfinite(train_x),train_x,med); sc=StandardScaler().fit(train_x); model=Ridge(alpha=ALPHA).fit(sc.transform(train_x),train_y)
  equal=(np.arange(len(pred),dtype=np.int64)*len(np.unique(months))//len(pred)).astype(np.int16)
  exact_x,_,exact_keys=summary(base[va],months); equal_x,_,equal_keys=summary(base[va],equal)
  exact_common=dict(zip(exact_keys,model.predict(sc.transform(np.where(np.isfinite(exact_x),exact_x,med))))); equal_common=dict(zip(equal_keys,model.predict(sc.transform(np.where(np.isfinite(equal_x),equal_x,med)))))
  scale=float(ally[tr].std()); exact=apply(pred,months,exact_common,scale); approx=apply(pred,equal,equal_common,scale); center=apply(pred,equal,{int(g):0.0 for g in np.unique(equal)},scale)
  mask=np.ones(len(months),bool) if name=='first' else (months>=62)&(months!=66)
  raw=cos(target[mask],pred[mask]); rows.append({'window':name,'raw':raw,'exact_common':cos(target[mask],exact[mask]),'equal_common':cos(target[mask],approx[mask]),'equal_center':cos(target[mask],center[mask]),'exact_delta':cos(target[mask],exact[mask])-raw,'equal_delta':cos(target[mask],approx[mask])-raw,'equal_vs_exact':cos(target[mask],approx[mask])-cos(target[mask],exact[mask])})
 out=pd.DataFrame(rows); out.to_csv(PROJECT/'outputs'/'deployable_equal_common_current138.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
