$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$pythonExe = 'D:\anaconda\envs\pytorch\python.exe'
$waitScript = Join-Path $PSScriptRoot 'wait_market_conditioned_event_residual_full_v28.py'
& $pythonExe -u $waitScript
exit $LASTEXITCODE
