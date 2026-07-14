@echo off
echo ===================================================
echo PlaceMux Phase 2 - Task 1
echo Marketplace Data Model and API Contract Validaton
echo ===================================================

echo.
echo Checking environment...
IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) ELSE (
    echo [INFO] No venv found, using system Python.
)

echo.
echo Running schema validation script...
python src\model_schemas.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Task 1 schemas validated successfully!
) else (
    echo.
    echo [ERROR] Task 1 failed during execution.
)

echo.
pause
