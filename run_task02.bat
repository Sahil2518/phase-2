@echo off
echo =======================================================
echo PlaceMux Phase 2 - Task 02: Match Vectors & Thresholds
echo =======================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [INFO] Running Match Vectors Pipeline...
python -m src.train_task02

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Task 02 completed successfully.
    echo Check logs/task02_metrics.json and logs/task02_matching_visualization.png
    echo =======================================================
) else (
    echo.
    echo =======================================================
    echo [ERROR] Pipeline failed with error code %ERRORLEVEL%.
    echo Check logs/task02_matching.log for details.
    echo =======================================================
)
echo.
pause
