@echo off
REM ============================================================
REM  PlaceMux — Task 07: AI/ML Monetization — Matching Tune
REM  One-click launcher
REM ============================================================

echo.
echo  ======================================================
echo   PlaceMux Task 07 — Matching Tune for Conversion
echo  ======================================================
echo.

REM Activate virtualenv if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [INFO] Running Task 07 Tuning Pipeline...
echo.

python -m src.train_task07

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ============================================================
    echo   SUCCESS — Matching Tune Pipeline complete.
    echo   Outputs:
    echo     models/ranker_v2_*                   ^<-- tuned model artifact
    echo     logs/task07_tuning_report.json       ^<-- delta report
    echo     logs/task07_tuning_chart.png
    echo     logs/task07.log
    echo  ============================================================
) else (
    echo.
    echo  ============================================================
    echo   FAILED — Tuning pipeline encountered errors.
    echo   Check logs/task07.log for details.
    echo  ============================================================
)

echo.
pause
