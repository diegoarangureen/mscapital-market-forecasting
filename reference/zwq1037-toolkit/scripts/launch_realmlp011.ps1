$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& 'D:\anaconda\envs\pytorch\python.exe' -u "$PSScriptRoot\exp_realmlp_011_our385_corr095.py"
exit $LASTEXITCODE
