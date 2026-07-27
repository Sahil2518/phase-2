@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 17: Placement Dashboards ^& Rec v1 Live
echo   Goal: Recommendation v1 live ^& demoable
echo ==============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

echo.
echo [STEP 1] Running Task 17 pipeline (Rec v1 + Dashboard)...
set PYTHONPATH=.
python src\train_task17.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAILED] Task 17 pipeline error. Check logs\task17.log
    pause
    exit /b 1
)

echo.
echo [STEP 2] Launching Rec v1 API on port 8001...
echo   Dashboard : http://localhost:8001/dashboard
echo   API Docs  : http://localhost:8001/docs
echo   Health    : http://localhost:8001/health
echo   Metrics   : http://localhost:8001/metrics
echo.
echo   Press Ctrl+C to stop the server.
echo.
python -m uvicorn src.rec_api:app --host 0.0.0.0 --port 8001 --reload

pause
