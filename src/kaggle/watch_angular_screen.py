"""Observe an X34 angular-loss screen and recover its Kaggle-side comparison.

No training launches, retries, or submissions. Local notifications require an
awake, connected desktop. Result calculation is independent of this watcher.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from audit.io import write_json
from watch_confirmation import publish, desktop_notice, status_name


def watch(args):
    from kaggle.api.kaggle_api_extended import KaggleApi
    repo=Path(__file__).resolve().parents[2]
    os.chdir(repo)
    api=KaggleApi();api.authenticate()
    hypothesis=getattr(args,'hypothesis','h1')
    if hypothesis not in ('h1','h2'):raise ValueError('Unknown hypothesis')
    kernel=f'diegoaranguren/mscapital-x34-{hypothesis}-tpu'
    status_path=repo/f'research/x34_{hypothesis}_status.json'
    report_path=repo/f'research/x34_{hypothesis}_review.json'
    output=repo/f'runs/x34-{hypothesis}'
    prefix='x34' if hypothesis=='h1' else 'x34_h2'
    expected_models=8 if hypothesis=='h1' else 4
    change={'angular_target':['noisy','clean']} if hypothesis=='h1' else {'angular_aggregation':['members','mean']}
    state={'kernel':kernel,'url':'https://www.kaggle.com/code/'+kernel,'version':1,
           'phase':'monitoring','submitted':False,'max_tpu_hours':3.5 if hypothesis=='h1' else 2,
           'monitor_pid':os.getpid(),'monitor_started_utc':datetime.now(timezone.utc).isoformat()}
    deadline=time.monotonic()+args.max_wait_hours*3600
    last=None;errors=0
    while time.monotonic()<deadline:
        try:
            status=status_name(api.kernels_status(kernel));errors=0
        except Exception as exc:
            errors+=1
            state.update(api_error_type=type(exc).__name__,consecutive_errors=errors)
            write_json(status_path,state)
            if errors>=10:
                state.update(phase='monitor_failed',needs_review=True);break
            time.sleep(60);continue
        state.update(kaggle_status=status,last_checked_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path,state)
        if status!=last:
            print(json.dumps({'status':status,'at':state['last_checked_utc']}),flush=True)
            if args.push:
                try:publish(repo,[status_path],f'Record X34 {hypothesis} status: '+status)
                except Exception as exc:state['git_publish_error']=type(exc).__name__
            last=status
        if status=='COMPLETE':
            try:
                api.kernels_output(kernel,str(output),file_pattern=rf'({prefix}_review\.json|{prefix}_progress\.json|run_worker_.*\.json|done\.json|predictions\.npz)$',quiet=True)
                progress=json.loads((output/f'{prefix}_progress.json').read_text())
                completed=len(progress['models']) if hypothesis=='h1' else progress['completed']
                if not progress['complete'] or completed!=expected_models:
                    raise ValueError('Incomplete panel; no promotion or automatic restart')
                result=json.loads((output/f'{prefix}_review.json').read_text())
                if not result['complete'] or result['changed_config']!=change:
                    raise ValueError('Unexpected comparison output')
                shutil.copyfile(output/f'{prefix}_review.json',report_path)
                state.update(phase='review_complete',models=expected_models,delta=result['paired']['delta'],
                             screen_gate_passed=result['gate']['passed'],
                             note='Screen only. No confirmation, next hypothesis, or submission launched automatically.')
            except Exception as exc:
                state.update(phase='review_failed',error_type=type(exc).__name__,needs_review=True)
            break
        if status in {'ERROR','FAILED','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'}:
            state.update(phase='remote_failed',needs_review=True)
            try:api.kernels_output(kernel,str(output),file_pattern=rf'({prefix}_progress\.json|run_worker_.*\.json)$',quiet=True)
            except Exception:pass
            break
        time.sleep(60)
    else:state.update(phase='monitor_timeout',needs_review=True)
    state['monitor_finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(status_path,state)
    if args.push:
        try:publish(repo,[status_path,report_path],f'Publish X34 {hypothesis} screening outcome')
        except Exception as exc:
            state['git_publish_error']=type(exc).__name__;write_json(status_path,state)
    if args.notify:
        desktop_notice(state['phase']=='review_complete',
                       f'Cribado X34 {hypothesis.upper()} finalizado o pendiente de revision. Consulta x34_{hypothesis}_status.json en GitHub. No se ha enviado submission.')
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hypothesis',choices=['h1','h2'],default='h1')
    p.add_argument('--max-wait-hours',type=float,default=12)
    p.add_argument('--push',action='store_true');p.add_argument('--notify',action='store_true')
    a=p.parse_args()
    if not 0<a.max_wait_hours<=24:p.error('Monitoring duration must be between zero and 24 hours')
    watch(a)
