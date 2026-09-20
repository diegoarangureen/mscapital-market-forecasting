"""Screen earlier sequence models as small Transformer-slot replacements."""
from __future__ import annotations
import json,os
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="2"
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/interim/tree_experiments/EXP-BLEND-006-EXTENDED-OWNED/validation_predictions.feather"
PARAM=ROOT/"data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE/fusion_parameters.npz"
OUT=ROOT/"outputs/submission_metadata/previous_sequence_slot_screen_20260920.json"
CANDIDATES={
"tsmixer":(ROOT/"data/interim/tree_experiments/EXP-SEQUENCE-021-TSMIXER-DEV/train059_valid6270_ex66/validation_predictions.feather","prediction"),
"linear_sequence":(ROOT/"data/interim/kaggle_results/linear_sequence_v23/linear_sequence_control/validation_predictions.csv","prediction"),
"last_readout":(ROOT/"data/interim/kaggle_results/transformer_last_readout_v22/factorized_transformer_last_readout/validation_predictions.csv","prediction"),
"sync_tcn":(ROOT/"data/interim/kaggle_outputs/multistream_sync_tcn_v17/synchronized_recent_tcn/validation_predictions.csv","prediction"),
"deep3_transformer":(ROOT/"data/interim/kaggle_outputs/multistream_factorized_transformer_deep3_dev/factorized_transformer/validation_predictions.csv","prediction"),
"transformer020_ema":(ROOT/"data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA/train059_valid6270_ex66/validation_predictions.feather","ema"),
"relative394_xs":(ROOT/"data/interim/sequence_experiments/EXP-TRANSFORMER-018-RELATIVE394-XS-CONFIRM-NO66/validation_predictions_epoch05.feather","prediction")}
def cos(y,p): return float(y@p/(np.linalg.norm(y)*np.linalg.norm(p)+1e-30))
def met(y,p,m):
 s=np.isin(m,[62,63,64,65]);f=np.isin(m,[67,68,69,70])
 return {"selection":cos(y[s],p[s]),"forward":cos(y[f],p[f]),"all":cos(y,p),
 "monthly_selection":{str(x):cos(y[m==x],p[m==x]) for x in range(62,66)},
 "monthly_forward":{str(x):cos(y[m==x],p[m==x]) for x in range(67,71)}}
def delta(c,b):
 return {"selection":c["selection"]-b["selection"],"forward":c["forward"]-b["forward"],"all":c["all"]-b["all"],
 "monthly_selection":{x:c["monthly_selection"][x]-b["monthly_selection"][x] for x in b["monthly_selection"]},
 "monthly_forward":{x:c["monthly_forward"][x]-b["monthly_forward"][x] for x in b["monthly_forward"]}}
def main():
 d=pd.read_feather(BASE);y=d.target.to_numpy(float);m=d.month.to_numpy();s=np.isin(m,[62,63,64,65])
 baseline=d.verified_joint.to_numpy(float);bm=met(y,baseline,m);q=np.load(PARAM)
 core=d[["tabm","realmlp","transformer","gru"]].to_numpy(float)/q["scale"].astype(float)
 v=np.nan_to_num(d.realized_volatility_60.to_numpy(float),nan=float(q["rv_median"]),posinf=float(q["rv_median"]),neginf=float(q["rv_median"]))
 gate=np.column_stack([np.interp(v,q["anchors"],q["state_weights"][:,j]) for j in range(4)])
 no_slot=(core*gate).sum(1)-gate[:,2]*core[:,2]
 rows=[]
 for name,(path,col) in CANDIDATES.items():
  x=pd.read_feather(path) if path.suffix==".feather" else pd.read_csv(path)
  x=x[["sample_id",col]].rename(columns={col:"candidate"})
  a=d[["sample_id"]].merge(x,on="sample_id",how="left",validate="one_to_one")
  assert a.candidate.notna().all(),name
  values=a.candidate.to_numpy(float);scale=float(np.sqrt(np.mean(values[s]**2)));z=values/max(scale,1e-12)
  full=no_slot+gate[:,2]*z
  variants=[]
  for units in range(13):
   weight=units*.025;p=(1-weight)*baseline+weight*full;cm=met(y,p,m);ch=delta(cm,bm)
   variants.append({"weight":weight,"metrics":cm,"delta":ch})
  best=max(variants,key=lambda r:r["metrics"]["selection"]);ch=best["delta"]
  passed=bool(best["weight"]>0 and ch["selection"]>=.0002 and ch["forward"]>=.0003 and ch["all"]>=.0003 and sum(x>=0 for x in ch["monthly_forward"].values())>=3 and min(ch["monthly_forward"].values())>=-.0015)
  rows.append({"name":name,"path":str(path),"column":col,"selection_rms":scale,"selected_weight":best["weight"],"metrics":best["metrics"],"delta_vs_verified_joint":ch,"passed":passed})
 report={"experiment":"EXP-BLEND-008-PREVIOUS-SEQUENCE-SLOT-SCREEN","selection_rule":"For each previous model choose 0..30% replacement in 2.5% steps using months62-65 only; months67-70 only gate.","baseline":bm,"rows":rows,"passed_models":[r["name"] for r in rows if r["passed"]],"formal_submission":False}
 OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
