@echo off
REM One-click setup + launch (Windows): venv -> dependencies -> data -> model -> dashboard
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating the virtual environment .venv ...
    python -m venv .venv || goto :error
)
set "PY=.venv\Scripts\python.exe"

echo [2/5] Installing dependencies ...
"%PY%" -m pip install --upgrade pip --quiet || goto :error
"%PY%" -m pip install -r requirements.txt --quiet || goto :error

echo [3/5] Generating 30 days of engine sensor data ...
"%PY%" dataset_builder.py || goto :error

echo [4/5] Training the 1 h / 4 h heat forecaster ...
"%PY%" train_model.py || goto :error

echo [5/5] Launching the dashboard on http://localhost:8501 (Ctrl+C to stop) ...
"%PY%" -m streamlit run app.py
goto :eof

:error
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
