@echo off
REM ============================================================
REM  PlaceMux — Task 04: Applications & Shortlisting
REM  One-click launcher
REM ============================================================

echo.
echo  ==========================================
echo   PlaceMux Task 04 — Match Explainability
echo  ==========================================
echo.

REM Activate virtualenv if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [INFO] Running Task 04 pipeline...
echo.

python -m src.train_task04

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ==========================================
    echo   SUCCESS — Task 04 completed.
    echo  ==========================================
) else (
    echo.
    echo  ==========================================
    echo   FAILED — Task 04 encountered an error.
    echo   Check logs/task04.log for details.
    echo  ==========================================
)

echo.
pause
