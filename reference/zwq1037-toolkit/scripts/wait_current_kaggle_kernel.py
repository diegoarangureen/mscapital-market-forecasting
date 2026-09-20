"""One-shot bounded wait for an already-running Kaggle kernel."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

CLI = r'D:\anaconda\envs\pytorch\Scripts\kaggle.exe'
parser = argparse.ArgumentParser()
parser.add_argument('--kernel', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--initial-delay', type=int, default=300)
args = parser.parse_args()


def pause(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(min(60, end-time.monotonic()))


pause(args.initial_delay)
deadline = time.monotonic() + 90*60
while time.monotonic() < deadline:
    try:
        auth = subprocess.run([CLI, 'auth', 'print-access-token'], capture_output=True,
                              text=True, timeout=90)
        environment = os.environ.copy()
        if auth.returncode == 0 and auth.stdout.strip():
            environment['KAGGLE_API_TOKEN'] = auth.stdout.strip()
        result = subprocess.run([CLI, 'kernels', 'status', args.kernel], env=environment,
                                capture_output=True, text=True, timeout=90)
        del environment, auth
        status = next((name for name in ('COMPLETE', 'ERROR', 'RUNNING', 'QUEUED')
                       if f'KernelWorkerStatus.{name}' in result.stdout), 'POLL_UNAVAILABLE')
    except subprocess.TimeoutExpired:
        status = 'POLL_UNAVAILABLE'
    report = {'kernel': args.kernel, 'status': status,
              'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if status in ('COMPLETE', 'ERROR'):
        break
    pause(300)
else:
    print('Observation deadline reached; job status is not terminal.', flush=True)
