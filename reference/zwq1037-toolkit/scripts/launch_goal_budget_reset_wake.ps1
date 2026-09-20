$ErrorActionPreference = 'Stop'
$startedAt = Get-Date
$wakeAt = $startedAt.AddHours(5)
Write-Output ("Goal budget-reset wake timer started at {0:o}; target {1:o}" -f $startedAt, $wakeAt)
Start-Sleep -Seconds 18000
Write-Output ("Five-hour goal wake timer completed at {0:o}" -f (Get-Date))
exit 0
