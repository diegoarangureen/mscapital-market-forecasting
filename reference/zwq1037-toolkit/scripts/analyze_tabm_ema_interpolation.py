import json
import numpy as np
import pandas as pd
p=r'data/interim/tree_experiments/EXP-TABM-044-MARKET385-K32-EMA/train059_valid6270_ex66/validation_predictions.feather'
x=pd.read_feather(p)
y=x.target.to_numpy(np.float64); raw=x.raw.to_numpy(np.float64); ema=x.ema.to_numpy(np.float64); m=x.month.to_numpy()
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
sel=(m>=62)&(m<=65); fwd=(m>=67)&(m<=70); allm=(m>=62)&(m<=70)&(m!=66)
rows=[]
for a in np.linspace(0,1,21):
 q=(1-a)*raw+a*ema
 rows.append({'ema_prediction_weight':round(float(a),2),'all_ex66':cos(y[allm],q[allm]),'selection_62_65':cos(y[sel],q[sel]),'forward_67_70':cos(y[fwd],q[fwd])})
best=max(rows,key=lambda z:z['selection_62_65'])
print(json.dumps({'best_by_selection':best,'raw':rows[0],'pure_ema':rows[-1],'grid':rows},ensure_ascii=False,indent=2))
