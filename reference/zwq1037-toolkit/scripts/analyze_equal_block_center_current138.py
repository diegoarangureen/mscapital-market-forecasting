"""Compare exact-month and equal-block centering on the current 0.138 OOF recipe."""
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_conv_hybrid_transformer_full_blend import load_window
from exp_tabm_014_quantile_stage_curves import unit
PROJECT=Path(__file__).resolve().parents[1]
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def center(pred,group):
 out=pred.copy()
 for g in np.unique(group):
  m=group==g; out[m]-=out[m].mean()
 return out
rows=[]
for name,fold,start,end in [('first','train049_valid5059',50,59),('second','train059_valid6070',60,70)]:
 current,parts,target,months,_=load_window(fold,start,end)
 trans=.335*parts['hybrid42']+.335*parts['hybrid137']+.33*parts['conv42']
 pred=.70*unit(current)+.30*unit(trans)
 n_groups=len(np.unique(months)); equal=(np.arange(len(pred),dtype=np.int64)*n_groups//len(pred)).astype(np.int16)
 exact=center(pred,months); approx=center(pred,equal)
 mask=np.ones(len(months),dtype=bool) if name=='first' else (months>=62)&(months!=66)
 rows.append({'window':name,'raw':cos(target[mask],pred[mask]),'exact_center':cos(target[mask],exact[mask]),'equal_center':cos(target[mask],approx[mask]),'exact_delta':cos(target[mask],exact[mask])-cos(target[mask],pred[mask]),'equal_delta':cos(target[mask],approx[mask])-cos(target[mask],pred[mask]),'equal_vs_exact':cos(target[mask],approx[mask])-cos(target[mask],exact[mask])})
out=pd.DataFrame(rows); out.to_csv(PROJECT/'outputs'/'equal_block_center_current138.csv',index=False); print(out.to_string(index=False))
