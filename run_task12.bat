@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 12: Resume/JD Parsing v0
echo   Goal: Produce structured skills from raw text
echo ==============================================================

if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No virtual environment found, using system Python.
)

echo [INFO] Running parsing v0 pipeline...
set PYTHONPATH=.
python src\train_task12.py

if %ERRORLEVEL% == 0 (
    echo.
    echo [SUCCESS] Task 12 pipeline completed successfully.
    echo [INFO] Parsed output saved to logs\task12_parsed_output.json
) else (
    echo.
    echo [FAILURE] Task 12 pipeline encountered an error. Check logs\task12.log.
)

pause
