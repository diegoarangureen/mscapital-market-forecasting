"""Conservative extension search around the verified raw40/event60 owned blend."""
from __future__ import annotations
import json, os
from pathlib import Path
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/interim/tree_experiments/EXP-BLEND-006-EXTENDED-OWNED/validation_predictions.feather"
PARAMETERS=ROOT/"data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE/fusion_parameters.npz"
OUT=ROOT/"data/interim/tree_experiments/EXP-BLEND-007-CONSERVATIVE-EXTENDED"
REPORT=ROOT/"outputs/submission_metadata/conservative_extended_owned_blend_search_20260920.json"
def cosine(y,p):
    return float(y@p/(np.linalg.norm(y)*np.linalg.norm(p)+1e-30))
def metrics(y,p,m):
    s=np.isin(m,[62,63,64,65]); f=np.isin(m,[67,68,69,70])
    return {"selection":cosine(y[s],p[s]),"forward":cosine(y[f],p[f]),"all":cosine(y,p),
        "monthly_selection":{str(x):cosine(y[m==x],p[m==x]) for x in range(62,66)},
        "monthly_forward":{str(x):cosine(y[m==x],p[m==x]) for x in range(67,71)}}
def deltas(c,b):
    return {"selection":c["selection"]-b["selection"],"forward":c["forward"]-b["forward"],"all":c["all"]-b["all"],
        "monthly_selection":{x:c["monthly_selection"][x]-b["monthly_selection"][x] for x in b["monthly_selection"]},
        "monthly_forward":{x:c["monthly_forward"][x]-b["monthly_forward"][x] for x in b["monthly_forward"]}}
def standardize(v,s):
    scale=float(np.sqrt(np.mean(v[s]**2))); return v/max(scale,1e-12),scale
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_feather(SOURCE)
    assert len(d)==140806 and d.sample_id.is_unique
    y=d.target.to_numpy(float); m=d.month.to_numpy(); s=np.isin(m,[62,63,64,65])
    base=d.verified_joint.to_numpy(float); bm=metrics(y,base,m)
    q=np.load(PARAMETERS); names=["tabm","realmlp","transformer","gru"]
    core=d[names].to_numpy(float)/q["scale"].astype(float)
    v=np.nan_to_num(d.realized_volatility_60.to_numpy(float),nan=float(q["rv_median"]),posinf=float(q["rv_median"]),neginf=float(q["rv_median"]))
    gate=np.column_stack([np.interp(v,q["anchors"],q["state_weights"][:,j]) for j in range(4)])
    old=core[:,2]; no_slot=(core*gate).sum(1)-gate[:,2]*old
    sub,sub_scale=standardize(d.subsecond.to_numpy(float),s)
    multi,multi_scale=standardize(d.multiwindow.to_numpy(float),s)
    sub_full=no_slot+gate[:,2]*sub; multi_full=no_slot+gate[:,2]*multi
    tree,tree_scale=standardize(d.tree_extra.to_numpy(float),s)
    tabm,tabm_scale=standardize(d.tabm_ema.to_numpy(float),s)
    rows=[]
    for su in range(11):
      for mu in range(11-su):
        sw=su*.025; mw=mu*.025
        if sw+mw>.25+1e-12: continue
        rj=1-sw-mw; slot=rj*base+sw*sub_full+mw*multi_full
        for tu in range(5):
          for au in range(5-tu):
            tw=tu*.025; aw=au*.025; ro=1-tw-aw
            p=ro*slot+tw*tree+aw*tabm; cm=metrics(y,p,m); ch=deltas(cm,bm)
            rows.append({"retained_joint_weight":rj,"subsecond_weight":sw,"multiwindow_weight":mw,
              "retained_owned_weight":ro,"tree_weight":tw,"tabm_ema_weight":aw,
              "selection":cm["selection"],"selection_delta":ch["selection"],
              "minimum_selection_month_delta":min(ch["monthly_selection"].values()),
              "nonnegative_selection_months":sum(x>=0 for x in ch["monthly_selection"].values()),
              "prediction":p,"metrics":cm,"delta":ch})
    stable=[r for r in rows if r["selection_delta"]>=.0003 and r["minimum_selection_month_delta"]>=-.0005 and r["nonnegative_selection_months"]>=3]
    z=max(stable if stable else rows,key=lambda r:r["selection"]); ch=z["delta"]
    passed=bool(ch["selection"]>=.0003 and ch["forward"]>=.0005 and ch["all"]>=.0005 and sum(x>=0 for x in ch["monthly_forward"].values())>=3 and min(ch["monthly_forward"].values())>=-.002)
    keys=("retained_joint_weight","subsecond_weight","multiwindow_weight","retained_owned_weight","tree_weight","tabm_ema_weight")
    report={"experiment":"EXP-BLEND-007-CONSERVATIVE-EXTENDED",
      "selection_rule":"Maximize months62-65 with Transformer additions capped at 25%, outer additions capped at 10%, at least 3/4 nonnegative selection months, and worst selection-month delta >= -0.0005.",
      "candidate_count":len(rows),"selection_stable_candidate_count":len(stable),
      "baseline_metrics":bm,"selected_weights":{k:z[k] for k in keys},"selected_metrics":z["metrics"],
      "delta_vs_verified_joint":ch,"scales":{"subsecond":sub_scale,"multiwindow":multi_scale,"tree":tree_scale,"tabm_ema":tabm_scale},
      "passed":passed,"gate":{"selection_delta_min":.0003,"forward_delta_min":.0005,"all_delta_min":.0005,"forward_months_nonnegative_min":3,"worst_forward_month_delta_min":-.002},"formal_submission":False}
    out=d[["sample_id","month","target","verified_joint"]].copy(); out["candidate"]=z["prediction"]; out.to_feather(OUT/"validation_predictions.feather")
    compact=[]
    for r in rows:
      item={k:v for k,v in r.items() if k not in ("prediction","metrics","delta")}
      item.update({"forward":r["metrics"]["forward"],"all":r["metrics"]["all"],"forward_delta":r["delta"]["forward"],"all_delta":r["delta"]["all"],"worst_forward_month_delta":min(r["delta"]["monthly_forward"].values()),"nonnegative_forward_months":sum(x>=0 for x in r["delta"]["monthly_forward"].values())})
      compact.append(item)
    pd.DataFrame(compact).sort_values("selection",ascending=False).to_csv(OUT/"weight_grid.csv",index=False)
    (OUT/"score_only.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()
