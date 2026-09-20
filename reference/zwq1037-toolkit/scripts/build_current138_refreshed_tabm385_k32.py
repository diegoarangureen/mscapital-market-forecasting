"""Refresh current138 with the validated TabM385 k32 while retaining tree diversity."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
RUN_NAME="current138_refreshed_tabm385_k32"
def unit(x):
    x=np.asarray(x,dtype=np.float64); return (x-x.mean())/x.std()
def main():
    components=pd.read_feather(PROJECT/'outputs/predictions/gru_embedding_seed_ensemble_fulltrain_test.feather')
    old_current=pd.read_csv(PROJECT/'outputs/submissions/gru_embedding_seed_ensemble_fulltrain.csv')
    new_tabm=pd.read_csv(PROJECT/'outputs/submissions/tabm_relative319_xs40_order20_market6_k32_seed42_fulltrain.csv')
    ids=old_current.sample_id.to_numpy()
    for frame in (components,new_tabm):
        if not np.array_equal(ids,frame.sample_id.to_numpy()): raise AssertionError('Component IDs do not align')
    refreshed_inner=unit(old_current.prediction)+0.675*(unit(new_tabm.prediction)-unit(components.tabm_prediction))
    prediction=0.80*unit(refreshed_inner)+0.20*unit(new_tabm.prediction)
    if not np.isfinite(prediction).all(): raise AssertionError('Invalid prediction')
    out=PROJECT/'outputs/submissions'/f'{RUN_NAME}.csv'
    pd.DataFrame({'sample_id':ids,'prediction':prediction}).to_csv(out,index=False)
    report={'run_name':RUN_NAME,'structure':{'refreshed_current135':0.80,'outer_tabm385_k32':0.20,'inner_tabm_replacement_weight':0.675,'tree':'retain old tree for diversity','gru':'retain old two-seed joint GRU pending Kaggle v9'},'local_62_70_no66':{'old_current138_recorded':0.157651,'candidate':0.1602189723683853,'delta':0.0025679723683853},'tree_decision':'tree379 improves old current135 alone, but reduces the k32-optimal current138 from 0.16021897 to 0.16019468 at its best tested weight; excluded from this blend','rows':len(ids),'prediction_std':float(prediction.std()),'output_path':str(out),'submission_status':'prepared_not_uploaded'}
    meta=PROJECT/'outputs/submission_metadata'/f'{RUN_NAME}.json'; meta.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()