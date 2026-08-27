@echo off
echo ============================================
echo Fast Bhagavad Gita QA - Enhanced Version
echo ============================================
echo.

cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON=..\.venv\Scripts\python.exe"
    echo Using virtual environment...
) else (
    set "PYTHON=python"
    echo No virtual environment found. Using system Python...
)

echo Starting the ENHANCED application...
echo Instant search + AI summaries!
echo.

"%PYTHON%" -m streamlit run fast_app.py --server.port 8503

if %errorlevel% neq 0 (
    echo.
    echo Failed to start. Make sure dependencies are installed:
    echo   pip install -r requirements.txt
)

echo.
echo Application stopped.
pause
