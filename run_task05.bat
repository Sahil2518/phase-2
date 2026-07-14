@echo off
REM ============================================================
REM  PlaceMux — Task 05: Marketplace Integration & Validation
REM  One-click launcher
REM ============================================================

echo.
echo  ======================================================
echo   PlaceMux Task 05 — Company Portal API Validation
echo  ======================================================
echo.

REM Activate virtualenv if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [INFO] Running Task 05 End-to-End Validation Suite...
echo.

python -m src.validate_task05

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ==========================================
    echo   SUCCESS — All validations passed.
    echo  ==========================================
) else (
    echo.
    echo  ==========================================
    echo   FAILED — Validation encountered errors.
    echo   Check logs/task05_validation.log
    echo  ==========================================
)

echo.
pause
