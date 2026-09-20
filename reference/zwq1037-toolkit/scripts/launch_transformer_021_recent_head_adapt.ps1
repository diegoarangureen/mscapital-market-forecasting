$ErrorActionPreference='Stop'
$env:OMP_NUM_THREADS='2';$env:MKL_NUM_THREADS='2';$env:OPENBLAS_NUM_THREADS='2';$env:MKL_THREADING_LAYER='SEQUENTIAL'
& 'D:\anaconda\envs\pytorch\python.exe' -u "$PSScriptRoot\exp_transformer_021_recent_head_adapt.py"
exit $LASTEXITCODE
