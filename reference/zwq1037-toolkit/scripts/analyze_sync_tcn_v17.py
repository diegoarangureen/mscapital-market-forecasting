from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path('.')
def cos(y,p):
 d=np.linalg.norm(y)*np.linalg.norm(p)
 return float(y@p/d) if d else 0.0
paths={
'old':P/'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv',
'multiwindow':P/'data/interim/kaggle_outputs/multistream_transformer_multiwindow_v14/factorized_transformer_multiwindow10/validation_predictions.csv',
'sync_tcn':P/'data/interim/kaggle_outputs/multistream_sync_tcn_v17/synchronized_recent_tcn/validation_predictions.csv',
}
frames={}
for n,p in paths.items():
 d=pd.read_csv(p)
 keep=['sample_id','prediction']
 if 'target' in d.columns: keep+=['target']
 if 'month' in d.columns: keep+=['month']
 frames[n]=d[keep].rename(columns={'prediction':n})
base=frames['old']
for n in ['multiwindow','sync_tcn']:
 base=base.merge(frames[n][['sample_id',n]],on='sample_id',validate='one_to_one')
if 'target' not in base:
 labels=pd.read_feather(P/'data/raw/label.feather',columns=['sample_id','month','target'])
 base=base.merge(labels,on='sample_id',validate='one_to_one')
base=base[base.month.ne(66)].copy()
y=base.target.to_numpy(float); m=base.month.to_numpy()
rows=[]
for n in ['old','multiwindow','sync_tcn']:
 p=base[n].to_numpy(float)
 rows.append({'name':n,'cosine':cos(y,p),'monthly':{str(int(x)):cos(y[m==x],p[m==x]) for x in np.unique(m)}})
old=base.old.to_numpy(float)
for n in ['multiwindow','sync_tcn']:
 q=base[n].to_numpy(float)
 for f in [.25,.5,.75,1.0]:
  p=(1-f)*old+f*q
  rows.append({'name':f'old_{n}_fraction_{f}','cosine':cos(y,p),'delta_vs_old':cos(y,p)-cos(y,old),'monthly_delta':{str(int(x)):cos(y[m==x],p[m==x])-cos(y[m==x],old[m==x]) for x in np.unique(m)},'correlation':float(np.corrcoef(old,q)[0,1])})
report={'rows':len(base),'window':'62-70 excluding66','results':rows}
out=P/'outputs/submission_metadata/sync_tcn_v17_comparison.json'
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
