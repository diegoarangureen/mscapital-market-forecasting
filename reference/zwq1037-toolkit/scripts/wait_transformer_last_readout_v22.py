"""Wait outside model turns; download compact results from the exact Kaggle version."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CLI=r'D:\anaconda\envs\pytorch\Scripts\kaggle.exe'
KERNEL='zwq1037/multistream-factorized-transformer-multiwindow-dev'
VERSION=22
OUT=ROOT/'data/interim/kaggle_results/transformer_last_readout_v22'
PLAN=ROOT/'outputs/submission_metadata/astra_sequential_plan_20260917.json'
def command(arguments):
    auth=subprocess.run([CLI,'auth','print-access-token'],capture_output=True,text=True,timeout=90)
    if auth.returncode:raise RuntimeError('OAuth refresh failed; credentials are not logged')
    env=os.environ.copy();env['KAGGLE_API_TOKEN']=auth.stdout.strip()
    result=subprocess.run([CLI,*arguments],env=env,capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError('Kaggle operation failed: '+result.stdout+result.stderr)
    return result.stdout

def main():
    print('Continuation plan: '+str(PLAN),flush=True)
    # 非模型进程低频检查远端；只在终态唤醒一次。
    # Poll remotely in a non-model process and wake once on a terminal state.
    time.sleep(40*60)
    deadline=time.monotonic()+4*60*60
    errors=0
    while time.monotonic()<deadline:
        try:
            text=command(['kernels','status',KERNEL]);errors=0
        except Exception:
            errors+=1
            if errors>=3:raise
            time.sleep(10*60);continue
        print(text.strip(),flush=True)
        if 'KernelWorkerStatus.ERROR' in text or 'KernelWorkerStatus.CANCEL' in text:
            raise RuntimeError('Remote training did not complete successfully; read continuation plan and repair once.')
        if 'KernelWorkerStatus.COMPLETE' in text:break
        time.sleep(10*60)
    else:raise RuntimeError('Remote wait deadline reached; inspect exact version before continuing.')
    print(command(['kernels','output',KERNEL+'/'+str(VERSION),'-p',str(OUT),'--file-pattern',r'(score_only|result)\.json$|validation_predictions\.csv$']),flush=True)
    folder=OUT/'factorized_transformer_last_readout'
    score=json.loads((folder/'score_only.json').read_text())
    candidate=pd.read_csv(folder/'validation_predictions.csv')
    baseline=pd.read_csv(ROOT/'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv')
    comparison=candidate.merge(baseline[['sample_id','prediction']],on='sample_id',how='left',validate='one_to_one',suffixes=('','_baseline'))
    assert len(comparison)==140806 and comparison.prediction_baseline.notna().all()
    mask=comparison.month.ge(67).to_numpy();y=comparison.target.to_numpy(float)
    def cosine(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
    old=cosine(y[mask],comparison.prediction_baseline.to_numpy(float)[mask])
    new=cosine(y[mask],comparison.prediction.to_numpy(float)[mask])
    eligible=score['delta']>=.0005 and new>=old
    score.update({'forward67_70_baseline':old,'forward67_70_candidate':new,'forward_delta':new-old,'eligible_for_full_training':eligible})
    (folder/'score_only.json').write_text(json.dumps(score,indent=2))
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    step=next(s for s in plan['steps'] if s['id']=='transformer_last_readout')
    step.update({'status':'validation_complete','score_file':str(folder/'score_only.json'),'eligible_for_full_training':eligible})
    plan['next_action']='Prepare full Transformer training in the same notebook using the winning readout and best epoch.' if eligible else 'Retain original Transformer. Verify the saved soft-gate CSV and conclude the three planned experiments; do not retrain rejected candidates.'
    PLAN.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(score),flush=True)
    print('NEXT: '+plan['next_action']+' Plan: '+str(PLAN),flush=True)
if __name__=='__main__':main()
