# serve_web.ps1 — 一鍵啟動 Echo Stream Web server + Cloudflare quick tunnel
#
# 用法（或直接雙擊根目錄的 serve_web.bat）：
#   scripts\serve_web.ps1            # STT/LLM/TTS 全真（--real）
#   scripts\serve_web.ps1 -Fake      # 全 fake（驗串流接收，不需 GPU）
#   scripts\serve_web.ps1 -Stop      # 停掉 server 與 tunnel
#
# 注意：quick tunnel 是匿名臨時隧道，網址每次都不同、不會出現在
# Cloudflare 主控台，閒置一段時間會被回收——過期就重跑一次。

param(
    [switch]$Stop,
    [switch]$Fake,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $env:TEMP "echo_stream_serve.pids"

# 相容 bat shim 的「serve_web.bat stop」寫法
if ($Rest -contains "stop") { $Stop = $true }

# 停掉所有 Echo Stream 相關行程。
# ⚠️ 不能只靠 PID 檔——手動起的 server（或別的工具起的）不在檔裡，
# 而 Windows 允許兩個行程同時 bind 同一個 port，殘留的舊 server 會
# 默默搶走流量（2026-08-21 踩過）。所以按 port 掃 + PID 檔雙管齊下。
function Stop-EchoProcesses {
    $killed = 0
    # 1) 佔著 8770 的，只要命令列含 echo_stream 就收
    Get-NetTCPConnection -LocalPort 8770 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$_" -ErrorAction SilentlyContinue
            if ($proc -and $proc.CommandLine -match "echo_stream") {
                Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
                $killed++
            }
        }
    # 2) PID 檔記錄的（tunnel 不在 8770 上，靠這裡收）
    if (Test-Path $PidFile) {
        Get-Content $PidFile | ForEach-Object {
            Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
        Remove-Item $PidFile -ErrorAction SilentlyContinue
    }
    return $killed
}

if ($Stop) {
    $n = Stop-EchoProcesses
    Write-Host "已停止（port 掃到 $n 個 server + PID 檔記錄的行程）。"
    exit 0
}

# 啟動前先清場——殘留的舊 server 會讓新 server bind 失敗（刻意的，見 _StrictServer）
Stop-EchoProcesses | Out-Null
Start-Sleep -Seconds 1

# Python：優先環境變數，否則猜同層的 U.E.P Core env（與 config.py 的同層猜測一致）
$Py = $env:ECHO_STREAM_PYTHON
if (-not $Py) {
    $Py = Join-Path (Split-Path -Parent $Root) "U.E.P-s-Core\env\Scripts\python.exe"
}
if (-not (Test-Path $Py)) {
    Write-Host "找不到 Python：$Py（可設環境變數 ECHO_STREAM_PYTHON 覆寫）"
    exit 1
}
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "找不到 cloudflared，請先安裝。"
    exit 1
}

$Mode = if ($Fake) { @() } else { @("--real") }
$Extra = @($Rest | Where-Object { $_ -and $_ -ne "stop" })
$ServeArgs = @("-m", "echo_stream", "serve") + $Mode + $Extra

Write-Host ("啟動 server（{0}）…" -f ($(if ($Fake) { "全 fake" } else { "--real，首次載入約 40-60 秒" })))
$Server = Start-Process -FilePath $Py -ArgumentList $ServeArgs -WorkingDirectory $Root -PassThru

# 等 server 就緒
for (; ; ) {
    Start-Sleep -Seconds 3
    if ($Server.HasExited) {
        Write-Host "✗ server 行程已退出——看它的視窗訊息。"
        exit 1
    }
    try {
        $Status = Invoke-RestMethod "http://127.0.0.1:8770/api/status" -TimeoutSec 2
        if ($Status.error) { Write-Host "✗ 管線載入失敗：$($Status.error)"; break }
        if ($Status.ready) { Write-Host ("✓ server 就緒（{0:n1}s）" -f $Status.load_seconds); break }
    } catch { }
}

# ⚠️ 關鍵：--config 指向空檔。不蓋掉的話 quick tunnel 會誤讀
# ~/.cloudflared/config.yml（mc tunnel）的 ingress，所有請求都掉進
# catch-all 404（2026-08-20 踩過）。
$EmptyCfg = Join-Path $env:TEMP "cf_empty.yml"
Set-Content -Path $EmptyCfg -Value ""
$CfLog = Join-Path $env:TEMP "cf_quick.log"
Remove-Item $CfLog -ErrorAction SilentlyContinue

$Tunnel = Start-Process -FilePath "cloudflared" -ArgumentList @(
    "tunnel", "--config", $EmptyCfg, "--url", "http://127.0.0.1:8770"
) -RedirectStandardError $CfLog -PassThru -WindowStyle Hidden

@($Server.Id, $Tunnel.Id) | Set-Content $PidFile

# 從 log 撈網址
$Url = $null
for ($i = 0; $i -lt 30 -and -not $Url; $i++) {
    Start-Sleep -Seconds 2
    if (Test-Path $CfLog) {
        $Match = Select-String -Path $CfLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" |
            Select-Object -First 1
        if ($Match) { $Url = $Match.Matches[0].Value }
    }
}
if (-not $Url) {
    Write-Host "✗ 拿不到 tunnel 網址，檢查 $CfLog"
    exit 1
}

# 邊緣要一點時間收斂，驗到通才報網址
Write-Host "驗證 tunnel 連通性…"
$Ok = $false
for ($i = 0; $i -lt 25 -and -not $Ok; $i++) {
    Start-Sleep -Seconds 3
    try {
        $Check = Invoke-RestMethod "$Url/api/status" -TimeoutSec 10
        if ($Check.ready) { $Ok = $true }
    } catch { }
}

try { Set-Clipboard -Value $Url } catch { }
Write-Host ""
Write-Host "════════════════════════════════════════════════════"
Write-Host "  $Url"
Write-Host ("  （已複製到剪貼簿{0}）" -f ($(if ($Ok) { "，已驗證連通" } else { "；連通驗證逾時，開開看再說" })))
Write-Host "════════════════════════════════════════════════════"
Write-Host ""
Write-Host "  ⚠ 此網址是公開的，測完請執行：serve_web.bat stop"
