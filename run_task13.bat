@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 13: Proctoring FP Reduction (Ship)
echo   Goal: FP count reduced vs Task 11 baseline
echo ==============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [STEP 1] Running Task 13 proctoring pipeline...
set PYTHONPATH=.
python src\train_task13.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAILED] Task 13 pipeline encountered an error. Check logs\task13.log
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Task 13 complete. Check logs\task13_metrics.json for results.
echo.
pause
