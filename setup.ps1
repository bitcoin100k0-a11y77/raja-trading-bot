# setup.ps1 — ANi's FX Bot MT5 Setup Script
# Run as Administrator: Right-click > Run with PowerShell
# This script sets up everything on a fresh Windows VPS

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ANi's FX Bot v2.0 - MT5 Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Please run this script as Administrator!" -ForegroundColor Red
    Write-Host "Right-click the script > 'Run with PowerShell' as Admin" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$BOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Bot directory: $BOT_DIR" -ForegroundColor Green

# ============================================
# STEP 1: Check Python
# ============================================
Write-Host ""
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow

$pythonExists = $false
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3\.\d+") {
        Write-Host "  Python found: $pyVersion" -ForegroundColor Green
        $pythonExists = $true
    }
} catch {}

if (-not $pythonExists) {
    Write-Host "  Python not found. Installing Python 3.11..." -ForegroundColor Yellow

    $installerUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"

    Write-Host "  Downloading Python..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

    Write-Host "  Installing Python (this takes a minute)..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    try {
        $pyVersion = python --version 2>&1
        Write-Host "  Python installed: $pyVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ERROR: Python installation failed. Install manually from python.org" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# ============================================
# STEP 2: Install Python packages
# ============================================
Write-Host ""
Write-Host "[2/5] Installing Python packages..." -ForegroundColor Yellow

pip install --upgrade pip 2>&1 | Out-Null
pip install -r "$BOT_DIR\requirements.txt" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Some packages may have failed. Check above." -ForegroundColor Yellow
}

# Verify MetaTrader5 installed
$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -match "OK") {
    Write-Host "  MetaTrader5 package: OK" -ForegroundColor Green
} else {
    Write-Host "  ERROR: MetaTrader5 package failed to install!" -ForegroundColor Red
    Write-Host "  Try: pip install MetaTrader5" -ForegroundColor Yellow
}

# ============================================
# STEP 3: Create data directory
# ============================================
Write-Host ""
Write-Host "[3/5] Creating data directory..." -ForegroundColor Yellow

$dataDir = "$BOT_DIR\data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
    Write-Host "  Created: $dataDir" -ForegroundColor Green
} else {
    Write-Host "  Already exists: $dataDir" -ForegroundColor Green
}

# ============================================
# STEP 4: Set environment variables
# ============================================
Write-Host ""
Write-Host "[4/5] Setting environment variables..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  I need your MT5 demo account details." -ForegroundColor Cyan
Write-Host "  (You can change these later in System > Environment Variables)" -ForegroundColor Gray
Write-Host ""

# Prompt for MT5 credentials
$mt5Login = Read-Host "  MT5 Login (account number)"
$mt5Password = Read-Host "  MT5 Password"
$mt5Server = Read-Host "  MT5 Server (e.g., ICMarketsSC-Demo)"
$mt5Symbol = Read-Host "  Gold symbol name (default: XAUUSD)"
if ([string]::IsNullOrWhiteSpace($mt5Symbol)) { $mt5Symbol = "XAUUSD" }

# Telegram (optional)
Write-Host ""
$telegramEnabled = Read-Host "  Enable Telegram alerts? (y/n)"
$telegramToken = ""
$telegramChatId = ""
if ($telegramEnabled -eq "y") {
    $telegramToken = Read-Host "  Telegram Bot Token"
    $telegramChatId = Read-Host "  Telegram Chat ID"
}

# Set system environment variables
Write-Host ""
Write-Host "  Setting environment variables..." -ForegroundColor Yellow

# MT5 Connection
[System.Environment]::SetEnvironmentVariable("MT5_LOGIN", $mt5Login, "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_PASSWORD", $mt5Password, "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_SERVER", $mt5Server, "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_SYMBOL", $mt5Symbol, "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_MAGIC", "20260402", "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_SLIPPAGE", "20", "Machine")
[System.Environment]::SetEnvironmentVariable("MT5_FILLING", "IOC", "Machine")

# Safety — LIVE_MODE OFF by default
[System.Environment]::SetEnvironmentVariable("LIVE_MODE", "false", "Machine")
[System.Environment]::SetEnvironmentVariable("DRY_RUN", "false", "Machine")

# Trading parameters
[System.Environment]::SetEnvironmentVariable("INITIAL_BALANCE", "10000", "Machine")
[System.Environment]::SetEnvironmentVariable("RISK_PERCENT", "1.0", "Machine")
[System.Environment]::SetEnvironmentVariable("MAX_TRADES_PER_DAY", "3", "Machine")
[System.Environment]::SetEnvironmentVariable("DAILY_LOSS_LIMIT", "4.0", "Machine")
[System.Environment]::SetEnvironmentVariable("MIN_SL_PIPS", "20.0", "Machine")
[System.Environment]::SetEnvironmentVariable("MIN_TP1_PIPS", "30.0", "Machine")
[System.Environment]::SetEnvironmentVariable("DATA_DIR", "$BOT_DIR\data", "Machine")
[System.Environment]::SetEnvironmentVariable("LOG_LEVEL", "INFO", "Machine")

# Telegram
if ($telegramEnabled -eq "y") {
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_ENABLED", "true", "Machine")
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", $telegramToken, "Machine")
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", $telegramChatId, "Machine")
} else {
    [System.Environment]::SetEnvironmentVariable("TELEGRAM_ENABLED", "false", "Machine")
}

Write-Host "  Environment variables set!" -ForegroundColor Green

# Refresh current session env vars
$env:MT5_LOGIN = $mt5Login
$env:MT5_PASSWORD = $mt5Password
$env:MT5_SERVER = $mt5Server
$env:MT5_SYMBOL = $mt5Symbol
$env:MT5_MAGIC = "20260402"
$env:MT5_SLIPPAGE = "20"
$env:MT5_FILLING = "IOC"
$env:LIVE_MODE = "false"
$env:DRY_RUN = "false"
$env:INITIAL_BALANCE = "10000"
$env:RISK_PERCENT = "1.0"
$env:MAX_TRADES_PER_DAY = "3"
$env:DAILY_LOSS_LIMIT = "4.0"
$env:DATA_DIR = "$BOT_DIR\data"

# ============================================
# STEP 5: Create auto-start tasks
# ============================================
Write-Host ""
Write-Host "[5/5] Creating auto-start scheduled tasks..." -ForegroundColor Yellow

# Remove existing tasks if they exist
Unregister-ScheduledTask -TaskName "ANi_FX_Bot" -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "ANi_MT5_AutoStart" -Confirm:$false -ErrorAction SilentlyContinue

# Bot auto-start task
$botAction = New-ScheduledTaskAction -Execute "python" -Argument "$BOT_DIR\bot\main.py" -WorkingDirectory $BOT_DIR
$botTrigger = New-ScheduledTaskTrigger -AtStartup -RandomDelay (New-TimeSpan -Seconds 30)
$botSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "ANi_FX_Bot" -Action $botAction -Trigger $botTrigger -Settings $botSettings -RunLevel Highest -Description "ANi's FX Bot - MT5 Trading" | Out-Null
Write-Host "  Bot auto-start task created" -ForegroundColor Green

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  IMPORTANT CHECKLIST:" -ForegroundColor Yellow
Write-Host "  [1] Make sure MT5 is installed and running" -ForegroundColor White
Write-Host "  [2] Log into your demo account in MT5" -ForegroundColor White
Write-Host "  [3] Enable: Tools > Options > Expert Advisors > Allow algo trading" -ForegroundColor White
Write-Host "  [4] Keep MT5 open (minimize, don't close)" -ForegroundColor White
Write-Host ""
Write-Host "  TO START THE BOT:" -ForegroundColor Yellow
Write-Host "  Double-click: start_bot.bat" -ForegroundColor White
Write-Host ""
Write-Host "  LIVE_MODE is OFF (shadow mode)." -ForegroundColor Green
Write-Host "  Bot will generate signals but NOT place real orders." -ForegroundColor Green
Write-Host "  When ready for live: set LIVE_MODE=true in env vars." -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to finish"
