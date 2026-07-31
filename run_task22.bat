@echo off
echo.
echo =======================================================
echo PlaceMux Phase 2 - Task 22: Drift Monitoring
echo =======================================================
echo.

set PYTHONPATH=%CD%
python src\train_task22.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ ERROR: Task 22 Drift Pipeline failed.
    echo Please check logs\task22.log for details.
) else (
    echo.
    echo ✅ SUCCESS: Task 22 Drift Pipeline completed successfully.
)

echo.
pause
