"""Prepare one soft-gated transfer candidate around the scored 0.152 blend."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_current151_lower_public_grid import PATHS
ROOT=Path(__file__).resolve().parents[1]
PARAM=ROOT/'data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE/fusion_parameters.npz'
OUT=ROOT/'outputs/submissions/current152_owned_softgate25_transfer.csv'
def main():
    incumbent=pd.read_csv(ROOT/'outputs/submissions/current151_theoretical_geometry_optimum.csv')
    ids=incumbent.sample_id.to_numpy();old_prediction=incumbent.prediction.to_numpy(float)
    meta=json.loads((ROOT/'outputs/submission_metadata/current151_theoretical_geometry_optimum.json').read_text())
    weights=meta['best_weights'];keys=['tabm','realmlp','transformer_old','gru']
    columns=[]
    for name in keys:
        frame=pd.read_csv(PATHS[name])
        assert np.array_equal(frame.sample_id.to_numpy(),ids)
        p=frame.prediction.to_numpy(float)
        assert np.isfinite(p).all()
        columns.append((p-p.mean())/p.std())
    z=np.column_stack(columns)
    state=np.load(PARAM);anchors=state['anchors'];state_weights=state['state_weights']
    rv=pd.read_feather(ROOT/'data/processed/test_market_microstructure_features.feather',columns=['sample_id','realized_volatility_60'])
    rv=pd.DataFrame({'sample_id':ids}).merge(rv,on='sample_id',how='left',validate='one_to_one')
    v=rv.realized_volatility_60.to_numpy(float)
    missing_rv=int((~np.isfinite(v)).sum())
    v=np.nan_to_num(v,nan=float(state['rv_median']),posinf=float(state['rv_median']),neginf=float(state['rv_median']))
    gate=np.column_stack([np.interp(v,anchors,state_weights[:,j]) for j in range(4)])
    assert np.allclose(gate.sum(axis=1),1)
    mass=sum(weights[name] for name in keys)
    old_owned=z@np.array([weights[name] for name in keys])
    new_owned=mass*(z*gate).sum(axis=1)
    delta=new_owned-old_owned
    # 维持现有38块共同项；只替换块内自有模型信号。
    # Preserve the existing 38-block common term and replace only owned signals.
    groups=np.arange(len(ids))*38//len(ids)
    for group in range(38):
        mask=groups==group;delta[mask]-=delta[mask].mean()
    prediction=old_prediction+delta
    assert len(ids)==647896 and np.isfinite(prediction).all()
    assert max(abs(delta[groups==g].mean()) for g in range(38))<1e-12
    pd.DataFrame({'sample_id':ids,'prediction':prediction}).to_csv(OUT,index=False)
    report={'status':'prepared_not_submitted','output_path':str(OUT),'public_weights':{name:weights[name] for name in ['gpu','tpu','yangq']},'owned_total_weight':mass,'method':'rv60 soft interpolation of nonnegative state weights with 75% fixed + 25% learned shrinkage','validation':'Owned single-checkpoint proxy: forward delta +0.0012673932, 3/4 months improve.','transfer_limitation':'Current full-test owned block uses temporal TabM/GRU and end57 RealMLP; validation fitting uses aligned 0-59 models. This CSV transfers the recipe and has no verified whole-blend LB improvement.','missing_rv_filled':missing_rv,'rows':len(ids),'correlation_with_incumbent':float(np.corrcoef(old_prediction,prediction)[0,1]),'group_common_preserved':True}
    (ROOT/'outputs/submission_metadata/current152_owned_softgate25_transfer.json').write_text(json.dumps(report,indent=2))
    plan_path=ROOT/'outputs/submission_metadata/astra_sequential_plan_20260917.json'
    plan=json.loads(plan_path.read_text())
    next(s for s in plan['steps'] if s['id']=='rv60_soft_gate').update(status='complete_csv_prepared',csv=str(OUT))
    plan['remote_wait_job_record']='outputs/submission_metadata/transformer_last_readout_v22_wait_job.json'
    plan['current_action']='Wait outside model turns for Transformer v22. On wake verify results and proceed with qualifying full training; otherwise finish plan.'
    plan_path.write_text(json.dumps(plan,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
