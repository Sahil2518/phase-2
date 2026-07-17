@echo off
REM ============================================================
REM  PlaceMux — Task 08: Receipts, Refunds & Reconciliation
REM  Spend-Quality Guardrail Demo
REM ============================================================

echo.
echo  ======================================================
echo   PlaceMux Task 08 — Spend-Quality Guardrail
echo  ======================================================
echo.

REM Activate virtualenv if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [INFO] Running Task 08 Demo...
echo.

python -m src.demo_task08

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ============================================================
    echo   SUCCESS — Demo completed.
    echo  ============================================================
) else (
    echo.
    echo  ============================================================
    echo   FAILED — Demo encountered errors.
    echo  ============================================================
)

echo.
pause
