"""Train a LightGBM linear-leaf model on the 379-feature late split."""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"

import lightgbm as lgb
import numpy as np
import pandas as pd

import exp_tabm_001_exp053r_features as tabm
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER20_COLUMNS
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features

PROJECT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="EXP-TREE-083-LGBM379-LINEAR-LEAF"
sys.path.insert(0,str(PROJECT/"data/interim/kaggle_kernels/relative319_xs_tabm_notebook"))
from run_extracted import TOP_FEATURES, add_relative_features as add_xs_features

def cosine(y,p):
    return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)))

def monthly_scores(y,p,m):
    return {str(int(month)):cosine(y[m==month],p[m==month]) for month in np.unique(m)}

def main():
    out=PROJECT/"data/interim/tree_experiments"/EXPERIMENT_ID/"train059_valid6270_no66"
    out.mkdir(parents=True,exist_ok=True)
    print("loading 379 features",flush=True)
    frame,base_cols,_,_=tabm.load_exp053r_data(PROJECT)
    add_relative_features(PROJECT,frame)
    months=frame.month.to_numpy(np.int16)
    selected=frame[TOP_FEATURES].to_numpy(np.float32)
    xs_values,xs_cols=add_xs_features(selected,months,TOP_FEATURES)
    frame=pd.concat([frame,pd.DataFrame(xs_values,columns=xs_cols,index=frame.index)],axis=1,copy=False)
    del selected,xs_values
    order=pd.read_feather(PROJECT/"data/processed/train_order_quote_position_features.feather",columns=["sample_id",*ORDER20_COLUMNS]).sort_values("sample_id").reset_index(drop=True)
    assert np.array_equal(frame.sample_id.to_numpy(),order.sample_id.to_numpy())
    for c in ORDER20_COLUMNS:
        frame[c]=order[c].to_numpy(np.float32)
    del order
    features=[*base_cols,*RELATIVE_COLUMNS,*xs_cols,*ORDER20_COLUMNS]
    assert len(features)==379 and len(set(features))==379
    train=months<=59
    valid=(months>=62)&(months<=70)&(months!=66)
    ids=frame.loc[valid,"sample_id"].to_numpy()
    y_train=frame.loc[train,"target"].to_numpy(np.float64)
    y_valid=frame.loc[valid,"target"].to_numpy(np.float64)
    valid_months=months[valid]
    print("materializing train and valid matrices",flush=True)
    x_train=frame.loc[train,features].to_numpy(dtype=np.float32,copy=True)
    x_valid=frame.loc[valid,features].to_numpy(dtype=np.float32,copy=True)
    del frame
    gc.collect()
    means=np.nanmean(x_train,axis=0,dtype=np.float64).astype(np.float32)
    scales=np.nanstd(x_train,axis=0,dtype=np.float64).astype(np.float32)
    means=np.where(np.isfinite(means),means,0).astype(np.float32)
    scales=np.where(np.isfinite(scales)&(scales>1e-8),scales,1).astype(np.float32)
    for start in range(0,x_train.shape[1],32):
        end=min(start+32,x_train.shape[1])
        x_train[:,start:end]=(x_train[:,start:end]-means[start:end])/scales[start:end]
        x_valid[:,start:end]=(x_valid[:,start:end]-means[start:end])/scales[start:end]
    target_scale=float(np.std(y_train))
    train_set=lgb.Dataset(x_train,label=y_train/target_scale,feature_name=features,free_raw_data=True)
    params={
      "objective":"regression","metric":"None","boosting_type":"gbdt",
      "linear_tree":True,"device_type":"cpu","tree_learner":"serial","num_threads":2,
      "learning_rate":0.03,"num_leaves":15,"max_depth":4,"min_data_in_leaf":500,
      "lambda_l2":10.0,"linear_lambda":100.0,"lambda_l1":0.0,
      "feature_fraction":0.8,"bagging_fraction":1.0,"bagging_freq":0,
      "max_bin":127,"seed":42,"feature_fraction_seed":42,
      "verbosity":-1,"force_col_wise":True,
    }
    print("training linear-leaf LightGBM",flush=True)
    started=time.perf_counter()
    model=lgb.train(params,train_set,num_boost_round=800,callbacks=[lgb.log_evaluation(period=100)])
    seconds=time.perf_counter()-started
    baseline=pd.read_feather(PROJECT/"data/interim/tree_experiments/EXP-TREE-077-RELATIVE319-XS40-ORDER20/train059_valid6270_no66/validation_predictions.feather")
    assert np.array_equal(ids,baseline.sample_id.to_numpy())
    base_pred=baseline.candidate_tree379.to_numpy(np.float64)
    checkpoint_scores={}
    predictions={}
    for iteration in (200,400,800):
        pred=np.asarray(model.predict(x_valid,num_iteration=iteration),np.float64)*target_scale
        predictions[iteration]=pred
        checkpoint_scores[str(iteration)]=cosine(y_valid,pred)
    best_iteration=max(checkpoint_scores,key=checkpoint_scores.get)
    best_pred=predictions[int(best_iteration)]
    base_score=cosine(y_valid,base_pred)
    result={
      "experiment_id":EXPERIMENT_ID,
      "train_months":"0-59","purged_months":"60-61",
      "validation_months":"62-70","excluded_validation_months":[66],
      "feature_count":379,"training_rows":int(train.sum()),"validation_rows":int(valid.sum()),
      "feature_scaling":"train mean/std; NaN preserved",
      "target_scaling":target_scale,"parameters":params,"training_seconds":seconds,
      "scores":{"baseline_xgb379":base_score,"checkpoint_cosine":checkpoint_scores,
                "best_iteration":int(best_iteration),"candidate_linear_leaf":checkpoint_scores[best_iteration],
                "delta":checkpoint_scores[best_iteration]-base_score},
      "monthly_baseline":monthly_scores(y_valid,base_pred,valid_months),
      "monthly_candidate":monthly_scores(y_valid,best_pred,valid_months),
    }
    result["improved_months"]=sum(result["monthly_candidate"][k]>result["monthly_baseline"][k] for k in result["monthly_baseline"])
    pd.DataFrame({"sample_id":ids,"month":valid_months,"target":y_valid,"baseline_xgb379":base_pred,"candidate_linear_leaf":best_pred}).to_feather(out/"validation_predictions.feather")
    model.save_model(str(out/"model.txt"),num_iteration=int(best_iteration))
    np.savez(out/"scaling.npz",mean=means,std=scales,target_scale=np.array([target_scale]))
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
