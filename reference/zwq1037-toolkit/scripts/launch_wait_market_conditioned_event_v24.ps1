$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& 'D:\anaconda\envs\pytorch\python.exe' -u "$PSScriptRoot\wait_market_conditioned_event_v24.py"
exit $LASTEXITCODE
