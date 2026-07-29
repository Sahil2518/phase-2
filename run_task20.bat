@echo off
REM run_task20.bat
REM One-click Windows launcher for Task 20: Rec Validation

echo ============================================================
echo Starting Task 20 Pipeline: Recommendation Quality Validation
echo ============================================================

REM Check if virtual environment exists and activate it if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the main pipeline script
python src\train_task20.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo [OK] Pipeline completed successfully.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo [ERROR] Pipeline failed with error code %ERRORLEVEL%.
    echo Check logs/task20.log for details.
    echo ============================================================
)

pause
