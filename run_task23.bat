@echo off
echo ===================================================
echo PlaceMux AI / ML Engineer
echo Task 23: MLOps Foundations (Registry + Feature Store)
echo ===================================================

if exist venv\Scripts\activate (
    echo Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo [WARNING] No virtual environment found at venv\. Using system Python.
)

echo.
echo Running MLOps Pipeline...
python -m src.train_task23

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ MLOps Pipeline completed successfully!
    echo Creating submission ZIP archive...
    python zip_task23.py
) else (
    echo.
    echo ❌ Pipeline failed with error code %ERRORLEVEL%. Check logs/task23_mlops.log for details.
)

echo.
pause
