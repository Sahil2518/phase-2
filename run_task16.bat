@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 16: College Portal - Rec v1 Design Demo
echo   Goal: Recommendation v1 design ready ^& demoable
echo ==============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [STEP 1] Running Task 16 Rec v1 pipeline...
set PYTHONPATH=.
python src\train_task16.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAILED] Task 16 pipeline encountered an error. Check logs\task16.log
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Task 16 complete. Rec v1 design is ready.
echo   Rec v1 report   : logs\task16_rec_v1_report.json
echo   Summary metrics : logs\task16_metrics.json
echo   Heatmap         : logs\task16_rec_heatmap.png
echo.
pause
