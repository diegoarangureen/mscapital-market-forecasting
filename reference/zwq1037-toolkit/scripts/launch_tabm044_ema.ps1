$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
& 'D:\anaconda\envs\pytorch\python.exe' -u "$PSScriptRoot\exp_tabm_044_market385_k32_ema.py"
exit $LASTEXITCODE
