@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 14: Parsing into Ontology
echo   Goal: Parsed skills feed the ontology
echo ==============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [STEP 1] Running Task 14 parsing-into-ontology pipeline...
set PYTHONPATH=.
python src\train_task14.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAILED] Task 14 pipeline encountered an error. Check logs\task14.log
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Task 14 complete. Ontology output in logs\task14_ontology_output.json
echo.
pause
