@echo off
echo.
echo =======================================================
echo PlaceMux Phase 2 - Task 21: Fairness Audit
echo =======================================================
echo.

set PYTHONPATH=%CD%
python src\train_task21.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ ERROR: Task 21 Fairness Audit failed.
    echo Please check logs\task21.log for details.
) else (
    echo.
    echo ✅ SUCCESS: Task 21 Fairness Audit completed successfully.
)

echo.
pause
