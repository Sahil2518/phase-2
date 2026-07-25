@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 15: Trust Layer Integration ^& Dry Run
echo   Goal: AI trust features signed off
echo ==============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [STEP 1] Running Task 15 trust layer pipeline...
set PYTHONPATH=.
python src\train_task15.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAILED] Task 15 pipeline encountered an error. Check logs\task15.log
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Task 15 complete.
echo   Sign-off report : logs\task15_trust_signoff.json
echo   Summary metrics : logs\task15_metrics.json
echo   Chart           : logs\task15_trust_chart.png
echo.
pause
