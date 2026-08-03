@echo off
echo ===================================================
echo   PlaceMux Task 25 - Go-Live Monitoring
echo ===================================================

REM Check if virtual environment exists and activate it if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

set PYTHONPATH=%CD%

python src\train_task25.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Task 25 pipeline failed. Check logs\task25.log for details.
    pause
    exit /b %ERRORLEVEL%
)

python zip_task25.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Packaging failed.
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Task 25 complete!
pause
