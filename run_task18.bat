@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 18: Admin Console ^& Review Queue
echo   Goal: Strengthen recommendation explainability
echo ==============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [STEP 1] Running Task 18 explainability pipeline...
set PYTHONPATH=.
python src\train_task18.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAILED] Task 18 pipeline error. Check logs\task18.log
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Task 18 complete. Admin Console is ready.
echo   Open logs\review_queue.html in your browser.
echo.
pause
