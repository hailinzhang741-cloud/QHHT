# 开机/登录后补推：今日已错过窗口且未成功推送的时段
# 用法: powershell -ExecutionPolicy Bypass -File .\scripts\catch_up_push.ps1

$WorkDir = "E:\QHHT"
$BatPath = "$WorkDir\scripts\run_multi_forecast.bat"
$Python  = "C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe"
$LogFile = "$WorkDir\data\logs\catch_up_push.log"

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

Set-Location $WorkDir
if (-not (Test-Path $BatPath)) {
    Write-Log "ERROR bat not found: $BatPath"
    exit 1
}

$missedJson = & $Python -c "from notify_state import missed_slots_today; import json; print(json.dumps(missed_slots_today()))"
try {
    $missed = @($missedJson | ConvertFrom-Json)
} catch {
    Write-Log "ERROR parse missed slots: $missedJson"
    exit 1
}

if (-not $missed -or $missed.Count -eq 0) {
    Write-Log "无需补推"
    exit 0
}

Write-Log "待补推时段: $($missed -join ', ')"
foreach ($slot in $missed) {
    Write-Log "补推开始 slot=$slot"
    & $BatPath $slot
    $code = $LASTEXITCODE
    Write-Log "补推结束 slot=$slot exit=$code"
}

Write-Log "补推完成"
