"""Observe X34 H1 and recover the comparison produced inside Kaggle.

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
    kernel='diegoaranguren/mscapital-x34-h1-tpu'
    status_path=repo/'research/x34_h1_status.json'
    report_path=repo/'research/x34_h1_review.json'
    output=repo/'runs/x34-h1'
    state={'kernel':kernel,'url':'https://www.kaggle.com/code/'+kernel,'version':1,
           'phase':'monitoring','submitted':False,'max_tpu_hours':3.5,
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
                try:publish(repo,[status_path],'Record X34 H1 status: '+status)
                except Exception as exc:state['git_publish_error']=type(exc).__name__
            last=status
        if status=='COMPLETE':
            try:
                api.kernels_output(kernel,str(output),file_pattern=r'(x34_review\.json|x34_progress\.json|run_worker_.*\.json|done\.json|predictions\.npz)$',quiet=True)
                progress=json.loads((output/'x34_progress.json').read_text())
                if not progress['complete'] or len(progress['models'])!=8:
                    raise ValueError('Incomplete panel; no promotion or automatic restart')
                result=json.loads((output/'x34_review.json').read_text())
                if not result['complete'] or result['changed_config']!={'angular_target':['noisy','clean']}:
                    raise ValueError('Unexpected comparison output')
                shutil.copyfile(output/'x34_review.json',report_path)
                state.update(phase='review_complete',models=8,delta=result['paired']['delta'],
                             screen_gate_passed=result['gate']['passed'],
                             note='Screen only. No confirmation, next hypothesis, or submission launched automatically.')
            except Exception as exc:
                state.update(phase='review_failed',error_type=type(exc).__name__,needs_review=True)
            break
        if status in {'ERROR','FAILED','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'}:
            state.update(phase='remote_failed',needs_review=True)
            try:api.kernels_output(kernel,str(output),file_pattern=r'(x34_progress\.json|run_worker_.*\.json)$',quiet=True)
            except Exception:pass
            break
        time.sleep(60)
    else:state.update(phase='monitor_timeout',needs_review=True)
    state['monitor_finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(status_path,state)
    if args.push:
        try:publish(repo,[status_path,report_path],'Publish X34 H1 screening outcome')
        except Exception as exc:
            state['git_publish_error']=type(exc).__name__;write_json(status_path,state)
    if args.notify:
        desktop_notice(state['phase']=='review_complete',
                       'Cribado X34 H1 finalizado o pendiente de revision. Consulta x34_h1_status.json en GitHub. No se ha enviado submission.')
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-wait-hours',type=float,default=12)
    p.add_argument('--push',action='store_true');p.add_argument('--notify',action='store_true')
    a=p.parse_args()
    if not 0<a.max_wait_hours<=24:p.error('Monitoring duration must be between zero and 24 hours')
    watch(a)
