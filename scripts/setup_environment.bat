@echo off
echo ========================================================
echo   Setting up Sentinel AI Environment
echo ========================================================

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.12 not found. Please install Python 3.12 from python.org.
    pause
    exit /b 1
)

echo [1/3] Upgrading pip...
py -3.12 -m pip install --upgrade pip

echo [2/3] Installing dependencies from requirements.txt...
py -3.12 -m pip install -r requirements.txt

echo [3/3] Preparing NASA C-MAPSS dataset...
py -3.12 scripts\download_cmapss.py

echo.
echo Setup completed successfully!
echo To train baseline models and start the platform, run:
echo   powershell -ExecutionPolicy Bypass -File scripts\run_all_local.ps1
echo.
pause
