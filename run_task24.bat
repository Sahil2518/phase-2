@echo off
REM run_task24.bat
REM One-click Windows launcher for Task 24: Fairness Audit

echo ===================================================
echo   PlaceMux Task 24 - Fairness Audit and Sign-off
echo ===================================================

REM Check if virtual environment exists and activate it if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python src\train_task24.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Task 24 pipeline failed. Check logs\task24.log for details.
    pause
    exit /b %ERRORLEVEL%
)

python zip_task24.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Packaging failed.
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Task 24 complete!
pause
