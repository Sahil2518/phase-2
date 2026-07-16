@echo off
REM ============================================================
REM  PlaceMux — Task 06: AI/ML Monetization — Match-Quality Baseline
REM  One-click launcher
REM ============================================================

echo.
echo  ======================================================
echo   PlaceMux Task 06 — Match-Quality Baseline Recorder
echo  ======================================================
echo.

REM Activate virtualenv if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [INFO] Running Task 06 Baseline Recording Pipeline...
echo.

python -m src.train_task06

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ============================================================
    echo   SUCCESS — Pre-monetization baseline recorded.
    echo   Outputs:
    echo     logs/task06_baseline.json  ^<-- immutable baseline record
    echo     logs/task06_baseline_chart.png
    echo     logs/task06.log
    echo  ============================================================
) else (
    echo.
    echo  ============================================================
    echo   FAILED — Baseline recording encountered errors.
    echo   Check logs/task06.log for details.
    echo  ============================================================
)

echo.
pause
