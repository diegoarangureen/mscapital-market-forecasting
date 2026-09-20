$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& 'D:\anaconda\envs\pytorch\python.exe' -u 'F:\深度学习\projects\mscapital_market_forecasting\scripts\exp_transformer_023_patch_transformer_gru.py'
exit $LASTEXITCODE
