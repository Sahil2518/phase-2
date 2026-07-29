@echo off
REM run_task19.bat
REM One-click Windows launcher for Task 19: Item-Bank Quality

echo ============================================================
echo Starting Task 19 Pipeline: Item-Bank Quality Evaluation
echo ============================================================

REM Check if virtual environment exists and activate it if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the main pipeline script
python src\train_task19.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo [OK] Pipeline completed successfully.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo [ERROR] Pipeline failed with error code %ERRORLEVEL%.
    echo Check logs/task19.log for details.
    echo ============================================================
)

pause
