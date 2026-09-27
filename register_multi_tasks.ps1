# Register CU+SC multi forecast tasks (Mon-Fri 08:40/10:30/14:15/20:50)
# Run as Administrator:
#   powershell -ExecutionPolicy Bypass -File .\scripts\register_multi_tasks.ps1

$OldTaskName = "QHHT_CU_DailyForecast"
$BatPath     = "E:\QHHT\scripts\run_multi_forecast.bat"
$WorkDir     = "E:\QHHT"

if (-not (Test-Path $BatPath)) {
    Write-Error "Bat not found: $BatPath"
    exit 1
}

foreach ($name in @($OldTaskName, "QHHT_MultiForecast_0840", "QHHT_MultiForecast_1030", "QHHT_MultiForecast_1415", "QHHT_MultiForecast_2050")) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
}
Write-Host "Removed old tasks including $OldTaskName"

$slots = @(
    @{ Time = "08:40"; Slot = "0840" },
    @{ Time = "10:30"; Slot = "1030" },
    @{ Time = "14:15"; Slot = "1415" },
    @{ Time = "20:50"; Slot = "2050" }
)

foreach ($s in $slots) {
    $subName = "QHHT_MultiForecast_$($s.Slot)"
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $s.Time
    $action  = New-ScheduledTaskAction -Execute $BatPath -Argument $s.Slot -WorkingDirectory $WorkDir
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -WakeToRun `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 2) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    Register-ScheduledTask -TaskName $subName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "Registered: $subName at $($s.Time) Mon-Fri (WakeToRun, Restart x3)"
}

# 登录后补推：PC 周末/休眠错过时段时自动补发
$CatchUpName = "QHHT_MultiForecast_CatchUp"
$CatchUpPs   = "E:\QHHT\scripts\catch_up_push.ps1"
Unregister-ScheduledTask -TaskName $CatchUpName -Confirm:$false -ErrorAction SilentlyContinue
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
$catchAction  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$CatchUpPs`"" -WorkingDirectory $WorkDir
$catchSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $CatchUpName -Action $catchAction -Trigger $logonTrigger -Settings $catchSettings -Force | Out-Null
Write-Host "Registered: $CatchUpName at Logon (missed slot catch-up)"

Write-Host "Test: scripts\run_multi_forecast.bat 0840"
Write-Host "Catch-up: powershell -ExecutionPolicy Bypass -File .\scripts\catch_up_push.ps1"
