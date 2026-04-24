# RUN_BOT.ps1 -- Right-click > Run with PowerShell
# ================================================
# EDIT THESE 3 VALUES FIRST TIME ONLY:

$MT5_LOGIN    = "YOUR_ACCOUNT_NUMBER"
$MT5_PASSWORD = "YOUR_PASSWORD"
$MT5_SERVER   = "YOUR_SERVER"

# TELEGRAM ALERTS (get from @BotFather on Telegram)
$TELEGRAM_TOKEN   = ""   # Example: 123456:ABC-DEF1234ghIkl-zyx57W2v
$TELEGRAM_CHAT_ID = ""   # Example: 987654321

# ================================================

$env:MT5_LOGIN=$MT5_LOGIN
$env:MT5_PASSWORD=$MT5_PASSWORD
$env:MT5_SERVER=$MT5_SERVER
$env:MT5_SYMBOL="XAUUSD"
$env:MT5_MAGIC="20260402"
$env:MT5_SLIPPAGE="20"
$env:MT5_FILLING="IOC"
$env:LIVE_MODE="true"
$env:DRY_RUN="false"
$env:INITIAL_BALANCE="10000"
$env:RISK_PERCENT="1.0"
$env:MAX_TRADES_PER_DAY="3"
$env:DAILY_LOSS_LIMIT="4.0"
$env:MIN_SL_PIPS="20.0"
$env:MIN_TP1_PIPS="30.0"
$env:SESSION_START="8"         # 08:00 UTC = London open. DO NOT set below 8.
$env:SESSION_END="22"          # 22:00 UTC = NY close
$env:BLACKOUT_START="11"       # 11:00-14:00 UTC blackout (NY open whipsaw)
$env:BLACKOUT_END="14"
$env:MAX_ENTRY_SLIPPAGE_PIPS="20"  # LIVE RISK -- skip if >20 pips past C0 close (30s window)
$env:UPDATE_INTERVAL="30"          # Tick loop interval in seconds (30 = default)
$env:DATA_DIR="$PSScriptRoot\data"
$env:LOG_LEVEL="INFO"
if ($TELEGRAM_TOKEN -ne "") {
    $env:TELEGRAM_ENABLED="true"
    $env:TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN
    $env:TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID
} else {
    $env:TELEGRAM_ENABLED="false"
}

if (-not (Test-Path "$PSScriptRoot\data")) { New-Item -ItemType Directory -Path "$PSScriptRoot\data" | Out-Null }

Write-Host "ANi's FX Bot starting..." -ForegroundColor Cyan
Write-Host "Login: $MT5_LOGIN | Server: $MT5_SERVER | Symbol: XAUUSD" -ForegroundColor Yellow
Write-Host "LIVE_MODE=true (demo trading)" -ForegroundColor Green
Write-Host ""

Set-Location $PSScriptRoot

try {
    python bot\main.py
} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Red
Write-Host "Bot stopped or crashed. Check error above." -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Read-Host "Press Enter to close"
