"""Build the aggressive direct-owned-block blend with the full-train time-aware GRU."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factorized_transformer_lb_candidates import build_common_component

PROJECT=Path(__file__).resolve().parents[1]
RUN_NAME="aggressive_direct_owned_block_gru_equal38_common"
WEIGHTS={
  "gpu_tabm142":0.20,
  "tpu_tabm142":0.12,
  "yangq_blend142":0.32,
  "tabm385_k32":0.126,
  "realmlp_corr095":0.054,
  "factorized_transformer":0.162,
  "timeaware_gru":0.018,
}
PATHS={
  "gpu_tabm142":PROJECT/"data/external/kaggle_public/bestwater_cos689/output/submission.csv",
  "tpu_tabm142":PROJECT/"data/external/kaggle_public/bestwater_tpu_v6/output/submission.csv",
  "yangq_blend142":PROJECT/"data/external/kaggle_public/yangq_lb142/output/submission.csv",
  "tabm385_k32":PROJECT/"outputs/submissions/tabm_relative319_xs40_order20_market6_k32_seed42_fulltrain.csv",
  "realmlp_corr095":PROJECT/"outputs/submissions/realmlp379_rq16_corr095_e8_fulltrain.csv",
  "factorized_transformer":PROJECT/"data/interim/kaggle_outputs/multistream_factorized_transformer_fulltrain/factorized_transformer/factorized_transformer379_full_e4.csv",
  "timeaware_gru":PROJECT/"data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_fulltrain/factorized_gru_fulltrain/factorized_gru379_full_e4.csv",
}

def zunit(x):
    x=np.asarray(x,dtype=np.float64)
    return (x-x.mean())/x.std()

def main():
    assert abs(sum(WEIGHTS.values())-1)<1e-12
    frames={k:pd.read_csv(p) for k,p in PATHS.items()}
    labels,template,groups,common=build_common_component()
    ids=template.sample_id.to_numpy()
    for name,frame in frames.items():
        assert list(frame.columns)==["sample_id","prediction"],(name,frame.columns.tolist())
        assert len(frame)==len(ids),(name,len(frame),len(ids))
        assert frame.sample_id.is_unique,name
        assert np.array_equal(frame.sample_id.to_numpy(),ids),name
        assert np.isfinite(frame.prediction.to_numpy(float)).all(),name
    source={k:v.prediction.to_numpy(float) for k,v in frames.items()}
    before=sum(WEIGHTS[k]*zunit(source[k]) for k in WEIGHTS)
    prediction=before.copy()
    target_scale=float(labels.target.to_numpy(float).std())
    for group in np.unique(groups):
        mask=groups==group
        prediction[mask]=prediction[mask]-prediction[mask].mean()+common[int(group)]/target_scale
    assert prediction.shape==(len(ids),) and np.isfinite(prediction).all()
    out=PROJECT/f"outputs/submissions/{RUN_NAME}.csv"
    no_common=PROJECT/f"outputs/submissions/{RUN_NAME}_no_common.csv"
    pd.DataFrame({"sample_id":ids,"prediction":prediction}).to_csv(out,index=False)
    pd.DataFrame({"sample_id":ids,"prediction":before}).to_csv(no_common,index=False)
    current_path=PROJECT/"outputs/submissions/factorized_full_replace_corr095_current138k32_equal38_common.csv"
    current=pd.read_csv(current_path)
    assert np.array_equal(current.sample_id.to_numpy(),ids)
    names=list(WEIGHTS)
    report={
      "run_name":RUN_NAME,
      "weights_before_common_component":WEIGHTS,
      "sources":{k:str(v) for k,v in PATHS.items()},
      "local_owned_block_selection":{
        "weights":{"tabm385_k32":0.35,"realmlp_corr095":0.15,"factorized_transformer":0.45,"timeaware_gru":0.05},
        "late_no66_cosine":0.16523339881894245,
        "tree_weight":0.0,
      },
      "correlation_matrix_names":names,
      "correlation_matrix":np.corrcoef([source[k] for k in names]).tolist(),
      "comparison":{
        "correlation_with_current_lb0150":float(np.corrcoef(prediction,current.prediction.to_numpy(float))[0,1]),
        "delta_std_vs_current_lb0150":float((prediction-current.prediction.to_numpy(float)).std()),
        "before_vs_after_common":float(np.corrcoef(before,prediction)[0,1]),
      },
      "prediction":{"rows":len(ids),"mean":float(prediction.mean()),"std":float(prediction.std()),"min":float(prediction.min()),"max":float(prediction.max()),"finite":True},
      "output_path":str(out),"no_common_output_path":str(no_common),
      "submission_status":"prepared_not_submitted",
    }
    meta=PROJECT/f"outputs/submission_metadata/{RUN_NAME}.json"
    meta.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
