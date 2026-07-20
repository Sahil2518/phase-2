@echo off
setlocal
echo ==============================================================
echo   PlaceMux Task 11: Proctoring Hardening
echo   Focus: False-Positive Reduction
echo ==============================================================

if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
    echo [INFO] Virtual environment activated.
) else (
    echo [INFO] No virtual environment found, using system Python.
)

echo [INFO] Running proctoring hardening pipeline...
python src\train_task11.py

if %ERRORLEVEL% == 0 (
    echo.
    echo [SUCCESS] Task 11 pipeline completed successfully.
    echo [INFO] Metrics saved to logs\task11_metrics.json
    echo [INFO] Model saved to models\proctor_classifier_v1.pkl
) else (
    echo.
    echo [FAILURE] Task 11 pipeline encountered an error. Check logs\task11.log.
)

pause
