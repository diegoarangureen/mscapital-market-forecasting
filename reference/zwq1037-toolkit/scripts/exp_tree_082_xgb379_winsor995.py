"""Test mild absolute-target winsorization on the 379-feature XGBoost."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"

import numpy as np
import pandas as pd

import exp_tabm_001_exp053r_features as tabm
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER20_COLUMNS
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_027_031_xgboost_capacity_regularization import fit_model, model_parameters

PROJECT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="EXP-TREE-082-XGB379-WINSOR995"
sys.path.insert(0,str(PROJECT/"data/interim/kaggle_kernels/relative319_xs_tabm_notebook"))
from run_extracted import TOP_FEATURES, add_relative_features as add_xs_features

def cosine(y,p):
    return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)))

def monthly_scores(y,p,m):
    return {str(int(month)):cosine(y[m==month],p[m==month]) for month in np.unique(m)}

def main():
    out=PROJECT/"data/interim/tree_experiments"/EXPERIMENT_ID/"train059_valid6270_no66"
    out.mkdir(parents=True,exist_ok=True)
    frame,base_cols,_,_=tabm.load_exp053r_data(PROJECT)
    add_relative_features(PROJECT,frame)
    months=frame.month.to_numpy(np.int16)
    selected=frame[TOP_FEATURES].to_numpy(np.float32)
    xs_values,xs_cols=add_xs_features(selected,months,TOP_FEATURES)
    frame=pd.concat([frame,pd.DataFrame(xs_values,columns=xs_cols,index=frame.index)],axis=1,copy=False)
    order=pd.read_feather(PROJECT/"data/processed/train_order_quote_position_features.feather",columns=["sample_id",*ORDER20_COLUMNS]).sort_values("sample_id").reset_index(drop=True)
    assert np.array_equal(frame.sample_id.to_numpy(),order.sample_id.to_numpy())
    for c in ORDER20_COLUMNS:
        frame[c]=order[c].to_numpy(np.float32)
    del order
    features=[*base_cols,*RELATIVE_COLUMNS,*xs_cols,*ORDER20_COLUMNS]
    assert len(features)==379 and len(set(features))==379
    train=months<=59
    valid=(months>=62)&(months<=70)&(months!=66)
    raw_y=frame.loc[train,"target"].to_numpy(np.float64)
    threshold=float(np.quantile(np.abs(raw_y),0.995))
    winsor_y=np.clip(raw_y,-threshold,threshold)
    params=model_parameters(n_estimators=800,colsample_bytree=0.8,device="cuda",n_jobs=2)
    model,seconds=fit_model(frame,features,train,pd.Series(winsor_y),params)
    ids=frame.loc[valid,"sample_id"].to_numpy()
    y=frame.loc[valid,"target"].to_numpy(np.float64)
    vm=months[valid]
    pred=np.asarray(model.predict(frame.loc[valid,features]),np.float64)
    baseline=pd.read_feather(PROJECT/"data/interim/tree_experiments/EXP-TREE-077-RELATIVE319-XS40-ORDER20/train059_valid6270_no66/validation_predictions.feather")
    assert np.array_equal(ids,baseline.sample_id.to_numpy())
    base_pred=baseline.candidate_tree379.to_numpy(np.float64)
    result={
      "experiment_id":EXPERIMENT_ID,
      "train_months":"0-59","purged_months":"60-61",
      "validation_months":"62-70","excluded_validation_months":[66],
      "feature_count":379,"training_rows":int(train.sum()),"validation_rows":int(valid.sum()),
      "target_transform":{"type":"symmetric_abs_winsor","quantile":0.995,"threshold":threshold,"centered_after_clip":False},
      "parameters":params,"training_seconds":seconds,
      "scores":{"baseline_xgb379":cosine(y,base_pred),"candidate_w995":cosine(y,pred)},
      "monthly_baseline":monthly_scores(y,base_pred,vm),
      "monthly_candidate":monthly_scores(y,pred,vm),
    }
    result["scores"]["delta"]=result["scores"]["candidate_w995"]-result["scores"]["baseline_xgb379"]
    result["improved_months"]=sum(result["monthly_candidate"][k]>result["monthly_baseline"][k] for k in result["monthly_baseline"])
    pd.DataFrame({"sample_id":ids,"month":vm,"target":y,"baseline_xgb379":base_pred,"candidate_w995":pred}).to_feather(out/"validation_predictions.feather")
    save_model_safely(model,out/"model.json")
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
