$ErrorActionPreference = 'Stop'
$startedAt = Get-Date
$wakeAt = $startedAt.AddHours(10)
Write-Output ("Goal ten-hour wake timer started at {0:o}; target {1:o}" -f $startedAt, $wakeAt)
Start-Sleep -Seconds 36000
Write-Output ("Ten-hour goal wake timer completed at {0:o}" -f (Get-Date))
exit 0
