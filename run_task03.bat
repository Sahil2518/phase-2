@echo off
echo ============================================================
echo  PlaceMux Phase 2 — Task 03: Search ^& Discovery
echo  AI-Powered Job ^& Candidate Ranking
echo ============================================================
echo.

REM Activate virtual environment if present
IF EXIST ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
) ELSE IF EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) ELSE (
    echo [WARNING] No virtual environment found — using system Python.
)

echo.
echo [INFO] Installing / verifying dependencies...
pip install -q -r requirements.txt

echo.
echo [INFO] Running Task 03 pipeline...
echo ------------------------------------------------------------
python -m src.train_task03

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo  SUCCESS: Task 03 pipeline completed.
    echo  Outputs:
    echo    - models\ranker_v1_*.pkl        (serialised AI model)
    echo    - logs\task03_metrics.json      (RMSE, R2, Spearman rho)
    echo    - logs\task03_rankings.json     (ranked results)
    echo    - logs\task03_ranking_heatmap.png
    echo    - logs\task03.log               (full run log)
    echo ============================================================
) ELSE (
    echo.
    echo ============================================================
    echo  FAILED: Task 03 pipeline exited with error code %ERRORLEVEL%.
    echo  Check logs\task03.log for details.
    echo ============================================================
)

echo.
pause
