$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& 'D:\anaconda\envs\pytorch\python.exe' -u "$PSScriptRoot\wait_transformer020_raw_full_v26.py"
exit $LASTEXITCODE
