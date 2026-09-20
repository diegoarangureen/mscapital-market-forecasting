$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& 'D:\anaconda\envs\pytorch\python.exe' -u '.\scripts\exp_transformer_024_axial_lob.py'
exit $LASTEXITCODE
