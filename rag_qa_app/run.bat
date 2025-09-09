@echo off
echo ============================================
echo RAG Q and A Application - Simple Launcher
echo ============================================
echo.

cd /d "%~dp0"

echo Starting the RAG Q and A Application...
echo.

REM Use the existing virtual environment
if exist "..\.venv\Scripts\python.exe" (
    echo Using virtual environment...
    "..\.venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo Virtual environment not found. Using system Python...
    python -m streamlit run app.py
)

if %errorlevel% neq 0 (
    echo.
    echo Error occurred. Trying alternative Python commands...
    py -m streamlit run app.py
)

echo.
echo Press any key to exit...
pause >nul
