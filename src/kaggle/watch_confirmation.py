"""Monitor one existing Kaggle job, recover/review its results and publish reports.

Never launches, retries, cancels or submits competition predictions. Credentials
come from the official Kaggle client outside the repo. Intended as a local
background process; it needs this computer awake and connected.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit.io import write_json


def status_name(response):
    status = response.status
    return getattr(status, 'name', str(status).split('.')[-1]).upper()


def publish(repo, paths, message):
    git = ['git', '-c', 'safe.directory=' + repo.as_posix(), '-C', str(repo)]
    names = [str(p.relative_to(repo)) for p in paths if p.exists()]
    subprocess.run([*git, 'add', '--', *names], check=True, capture_output=True)
    changed = subprocess.run([*git, 'diff', '--cached', '--quiet', '--', *names]).returncode
    if changed == 1:
        subprocess.run([*git, 'commit', '--only', '-m', message, '--', *names], check=True, capture_output=True)
    elif changed != 0:
        raise RuntimeError('Git diff failed')
    subprocess.run([*git, 'push', 'origin', 'main'], check=True, capture_output=True, timeout=90)


def desktop_notice(success):
    if os.name != 'nt':
        return
    message = ('Confirmacion completa. Revisa el informe en GitHub. No se ha enviado ninguna submission.'
               if success else 'La confirmacion requiere revision. Consulta seed123_status.json en GitHub.')
    command = ("Add-Type -AssemblyName System.Windows.Forms; "
               "$msNotice = New-Object System.Windows.Forms.NotifyIcon; "
               "$msNotice.Icon = [System.Drawing.SystemIcons]::Information; "
               "$msNotice.Visible = $true; "
               f"$msNotice.ShowBalloonTip(15000, 'MSCapital', '{message}', [System.Windows.Forms.ToolTipIcon]::Info); "
               "Start-Sleep -Seconds 15; $msNotice.Dispose()")
    subprocess.Popen(['powershell', '-NoProfile', '-STA', '-WindowStyle', 'Hidden', '-Command', command],
                     creationflags=subprocess.CREATE_NO_WINDOW)


def watch(args):
    from kaggle.api.kaggle_api_extended import KaggleApi
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    api = KaggleApi(); api.authenticate()
    state_path = repo / 'research' / 'seed123_status.json'
    report_path = repo / 'research' / 'seed123_review.json'
    recovered = repo / 'runs' / 'seed123'
    output = repo / 'runs' / 'candidate-no-clip-third-seed'
    state = {'kernel': args.kernel, 'url': 'https://www.kaggle.com/code/' + args.kernel,
             'platform_timeout_hours': 3, 'submitted': False, 'monitor_pid': os.getpid(),
             'monitor_started_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'monitoring'}
    start, previous, errors = time.monotonic(), None, 0
    while time.monotonic() - start < args.max_wait_hours * 3600:
        try:
            response = api.kernels_status(args.kernel)
            status = status_name(response)
            errors = 0
        except Exception as exc:
            errors += 1
            state.update(api_error_type=type(exc).__name__, consecutive_errors=errors)
            write_json(state_path, state)
            if errors >= 10:
                state.update(phase='monitor_failed', needs_review=True)
                break
            time.sleep(60)
            continue
        state.update(kaggle_status=status, last_checked_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        if status != previous:
            print(json.dumps({'status': status, 'at': state['last_checked_utc']}), flush=True)
            if args.push:
                try:
                    publish(repo, [state_path], 'Record third-seed confirmation status: ' + status)
                    state.pop('git_publish_error', None)
                except Exception as exc:
                    state['git_publish_error'] = type(exc).__name__
            previous = status
        if status == 'COMPLETE':
            state['phase'] = 'recovering_results'; write_json(state_path, state)
            try:
                api.kernels_output(args.kernel, str(recovered),
                    file_pattern=r'(predictions\.npz|done\.json|run_worker_.*\.json|summary\.json)$', quiet=True)
                summary = json.loads((recovered / 'run' / 'summary.json').read_text())
                if not summary.get('complete'):
                    raise ValueError('Incomplete panel at budget stop; no automatic resume')
                cmd = [sys.executable, str(repo / 'src/kaggle/review_champion.py'),
                       '--labels', 'runs/recovered/labels', '--confirm', 'runs/recovered/confirm/run',
                       str(recovered / 'run'), '--final', 'runs/recovered/final/run',
                       '--template', 'runs/recovered/template/submission.csv', '--out', str(output)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                (recovered / 'review_stdout.log').write_text(result.stdout, encoding='utf-8')
                (recovered / 'review_stderr.log').write_text(result.stderr, encoding='utf-8')
                if result.returncode:
                    raise RuntimeError('Result verification failed; see local review_stderr.log')
                shutil.copyfile(output / 'review.json', report_path)
                review = json.loads(report_path.read_text())
                c = review['confirmation']
                state.update(phase='review_complete', models=c['models'], seeds=c['seeds'],
                             no_clip_delta=c['no_clip_minus_champion']['delta'],
                             note='Reused validation windows. Diego decides submission; no LB claim.')
            except Exception as exc:
                state.update(phase='review_failed', error_type=type(exc).__name__, needs_review=True)
            break
        if status in {'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_ACKNOWLEDGED', 'FAILED'}:
            state.update(phase='remote_failed', needs_review=True)
            break
        time.sleep(60)
    else:
        state.update(phase='monitor_timeout', needs_review=True,
                     note='No relaunch/cancellation. Remote job may still be queued/running.')
    state['monitor_finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    if args.push:
        try:
            publish(repo, [state_path, report_path], 'Publish third-seed confirmation outcome')
        except Exception as exc:
            state['git_publish_error'] = type(exc).__name__
            write_json(state_path, state)
    if args.notify:
        desktop_notice(state['phase'] == 'review_complete')
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', default='diegoaranguren/mscapital-confirm-seed123')
    p.add_argument('--max-wait-hours', type=float, default=6)
    p.add_argument('--push', action='store_true'); p.add_argument('--notify', action='store_true')
    a = p.parse_args()
    if not 0 < a.max_wait_hours <= 12:
        p.error('Monitoring duration must be between zero and twelve hours')
    watch(a)
