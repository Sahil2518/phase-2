@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 09: Failure Handling ^& Resilience Check
echo ==============================================================

if exist .venv\Scripts\activate (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate
) else if exist venv\Scripts\activate (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo [WARNING] No virtual environment found. Running globally.
)

set PYTHONPATH=%cd%
echo [INFO] Running validation pipeline...
python src\train_task09.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline failed!
) else (
    echo [SUCCESS] Pipeline completed successfully!
)

echo.
pause
