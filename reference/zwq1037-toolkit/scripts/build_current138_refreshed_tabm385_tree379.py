"""Refresh current138 by replacing its inner TabM and tree with validated models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
RUN_NAME="current138_refreshed_tabm385_tree379"
def unit(x):
    x=np.asarray(x,dtype=np.float64); return (x-x.mean())/x.std()
def main():
    components=pd.read_feather(PROJECT/'outputs/predictions/gru_embedding_seed_ensemble_fulltrain_test.feather')
    old_current=pd.read_csv(PROJECT/'outputs/submissions/gru_embedding_seed_ensemble_fulltrain.csv')
    new_tabm=pd.read_csv(PROJECT/'outputs/submissions/tabm_relative319_xs40_order20_market6_seed42_fulltrain.csv')
    new_tree=pd.read_csv(PROJECT/'outputs/submissions/xgboost_relative319_xs40_order20_fulltrain.csv')
    ids=old_current.sample_id.to_numpy()
    for frame in (components,new_tabm,new_tree):
        if not np.array_equal(ids,frame.sample_id.to_numpy()):
            raise AssertionError("Component IDs do not align")
    prediction=(
        unit(old_current.prediction)
        + 0.675*(unit(new_tabm.prediction)-unit(components.tabm_prediction))
        + 0.180*(unit(new_tree.prediction)-unit(components.old_tree_prediction))
    )
    prediction=unit(prediction)
    if not np.isfinite(prediction).all(): raise AssertionError("Invalid prediction")
    out=PROJECT/'outputs/submissions'/f'{RUN_NAME}.csv'
    pd.DataFrame({'sample_id':ids,'prediction':prediction}).to_csv(out,index=False)
    report={
      'run_name':RUN_NAME,
      'derivation':'saved current135 plus exact inner component replacement corrections',
      'replacements':{'inner_tabm_weight':0.675,'old_tabm':'relative319 quantile TabM','new_tabm':'TabM385 seed42','inner_tree_weight':0.18,'old_tree':'EXP053R tree','new_tree':'XGBoost379 Order20'},
      'outer_duplicate_tabm_weight':0.0,
      'local_62_70_no66':{'old_current135':0.15560598263783806,'old_current138_recorded':0.157651,'tabm_only_refresh':0.15996036120661758,'tree_only_refresh':0.15657801676188018,'both_refresh':0.16000772386442927,'delta_vs_old_current138_recorded':0.00235672386442927},
      'rows':len(ids),'prediction_mean':float(prediction.mean()),'prediction_std':float(prediction.std()),'output_path':str(out),'submission_status':'prepared_not_uploaded'
    }
    meta=PROJECT/'outputs/submission_metadata'/f'{RUN_NAME}.json'; meta.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()