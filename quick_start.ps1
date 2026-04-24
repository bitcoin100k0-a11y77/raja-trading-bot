# quick_start.ps1 — ANi's FX Bot - Quick Start
# =============================================
# EDIT THE 4 VALUES BELOW, THEN RUN THIS SCRIPT
# Right-click > Run with PowerShell (as Admin)
# =============================================

# ========== REPLACE THESE 4 VALUES ==========
$MT5_LOGIN    = "YOUR_MT5_ACCOUNT_NUMBER"     # Example: 12345678
$MT5_PASSWORD = "YOUR_MT5_PASSWORD"           # Example: abc123xyz
$MT5_SERVER   = "YOUR_MT5_SERVER_NAME"        # Example: ICMarketsSC-Demo
$MT5_SYMBOL   = "XAUUSD"                     # Change ONLY if your broker uses different name like GOLD
# =============================================

# Optional: Telegram alerts (leave empty to skip)
$TELEGRAM_TOKEN   = ""    # Example: 123456:ABC-DEF1234ghIkl-zyx57W2v
$TELEGRAM_CHAT_ID = ""    # Example: 987654321

# ========== DO NOT EDIT BELOW THIS LINE ==========

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ANi's FX Bot - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$BOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Step 1: Refresh PATH (in case Python was just installed)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "`n[1/4] Checking Python..." -ForegroundColor Yellow
try {
    $v = python --version 2>&1
    Write-Host "  $v" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found! Install from python.org first." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

# Step 2: Install packages
Write-Host "`n[2/4] Installing packages..." -ForegroundColor Yellow
pip install --upgrade pip 2>&1 | Out-Null
pip install MetaTrader5 pandas numpy requests python-telegram-bot 2>&1

$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -match "OK") {
    Write-Host "  MetaTrader5 package: OK" -ForegroundColor Green
} else {
    Write-Host "  ERROR: MetaTrader5 failed to install!" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

# Step 3: Create data folder
Write-Host "`n[3/4] Setting up..." -ForegroundColor Yellow
$dataDir = "$BOT_DIR\data"
if (-not (Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir | Out-Null }

# Step 4: Set ALL environment variables
Write-Host "`n[4/4] Setting environment variables..." -ForegroundColor Yellow

# MT5 Connection
[System.Environment]::SetEnvironmentVariable("MT5_LOGIN",    $MT5_LOGIN,    "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_PASSWORD",  $MT5_PASSWORD, "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_SERVER",    $MT5_SERVER,   "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_SYMBOL",    $MT5_SYMBOL,   "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_MAGIC",     "20260402",    "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_SLIPPAGE",  "20",          "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_FILLING",   "IOC",         "Machine")

# LIVE_MODE = true (demo account trading)
[System.Environment]::SetEnvironmentVariable("LIVE_MODE",     "true",        "Machine")
[System.Environment]::SetEnvironmentVariable("DRY_RUN",       "false",       "Machine")

# Risk settings
[System.Environment]::SetEnvironmentVariable("INITIAL_BALANCE",   "10000",      "Machine")
[System.Environment]::SetEnvironmentVariable("RISK_PERCENT",      "1.0",        "Machine")
[System.Environment]::SetEnvironmentVariable("MAX_TRADES_PER_DAY","3",          "Machine")
[System.Environment]::SetEnvironmentVariable("DAILY_LOSS_LIMIT",  "4.0",        "Machine")
[System.Environment]::SetEnvironmentVariable("MIN_SL_PIPS",       "20.0",       "Machine")
[System.Environment]::SetEnvironmentVariable("MIN_TP1_PIPS",      "30.0",       "Machine")
[System.Environment]::SetEnvironmentVariable("DATA_DIR",          "$BOT_DIR\data","Machine")
[System.Environment]::SetEnvironmentVariable("LOG_LEVEL",         "INFO",       "Machine")

# Telegram
if ($TELEGRAM_TOKEN -ne "") {
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_ENABLED",   "true",           "Machine")
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", $TELEGRAM_TOKEN,  "Machine")
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID",   $TELEGRAM_CHAT_ID,"Machine")
    Write-Host "  Telegram: ENABLED" -ForegroundColor Green
} else {
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_ENABLED",   "false",          "Machine")
    Write-Host "  Telegram: DISABLED (no token provided)" -ForegroundColor Gray
}

# Load into current session too
$env:MT5_LOGIN=$MT5_LOGIN; $env:MT5_PASSWORD=$MT5_PASSWORD; $env:MT5_SERVER=$MT5_SERVER
$env:MT5_SYMBOL=$MT5_SYMBOL; $env:MT5_MAGIC="20260402"; $env:MT5_SLIPPAGE="20"
$env:MT5_FILLING="IOC"; $env:LIVE_MODE="true"; $env:DRY_RUN="false"
$env:INITIAL_BALANCE="10000"; $env:RISK_PERCENT="1.0"; $env:MAX_TRADES_PER_DAY="3"
$env:DAILY_LOSS_LIMIT="4.0"; $env:DATA_DIR="$BOT_DIR\data"; $env:LOG_LEVEL="INFO"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "  LIVE_MODE = TRUE (demo trading ON)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Make sure MT5 is OPEN and LOGGED IN before continuing." -ForegroundColor Yellow
Write-Host ""
$start = Read-Host "  Start the bot now? (y/n)"
if ($start -eq "y") {
    Write-Host "`n  Starting bot..." -ForegroundColor Cyan
    Set-Location $BOT_DIR
    python bot\main.py
} else {
    Write-Host "`n  To start later, run: python bot\main.py" -ForegroundColor White
}
