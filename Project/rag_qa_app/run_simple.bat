@echo off
echo ============================================
echo RAG Q&A Application - Simple Runner
echo ============================================
echo.

cd /d "%~dp0"

echo Starting RAG Q&A Application...
echo Working directory: %CD%
echo.

REM Check if the main virtual environment exists
if exist "..\..\.venv\Scripts\python.exe" (
    echo ✅ Using project virtual environment...
    "..\..\.venv\Scripts\python.exe" -m streamlit run app.py
) else if exist "rag_env\Scripts\python.exe" (
    echo ✅ Using local virtual environment...
    "rag_env\Scripts\python.exe" -m streamlit run app.py
) else (
    echo ⚠️ No virtual environment found. Trying system Python...
    python -m streamlit run app.py
    if %errorlevel% neq 0 (
        echo Trying with 'py' command...
        py -m streamlit run app.py
    )
)

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to start the application.
    echo Please make sure Python and required packages are installed.
    echo.
    echo You can install requirements manually:
    echo pip install -r requirements.txt
    echo.
)

echo.
echo Press any key to exit...
pause >nul
