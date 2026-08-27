@echo off
echo ================================================================
echo                RAG Q and A Application
echo        One-Click Setup and Launch
echo ================================================================
echo.

cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON=..\.venv\Scripts\python.exe"
    echo Using virtual environment...
) else (
    set "PYTHON=python"
    echo No virtual environment found. Using system Python...
)

echo.
echo Opening http://localhost:8502 in your browser...
echo.

"%PYTHON%" -m streamlit run app.py --server.port 8502

if %errorlevel% neq 0 (
    echo.
    echo Failed to start. Make sure dependencies are installed:
    echo   pip install -r requirements.txt
)

echo.
echo Application stopped.
pause
