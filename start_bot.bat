@echo off
echo ============================================
echo   ANi's FX Bot v2.0 - MT5 Edition
echo   Starting...
echo ============================================
echo.

cd /d "%~dp0"
python bot\main.py

echo.
echo Bot stopped. Press any key to exit.
pause > nul
