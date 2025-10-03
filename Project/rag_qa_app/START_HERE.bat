@echo off
echo ================================================================
echo                RAG Q and A Application
echo        One-Click Setup and Launch
echo ================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking virtual environment...
if not exist "..\.venv\Scripts\python.exe" (
    echo Virtual environment not found! Please run setup first.
    echo.
    echo To set up, open PowerShell as Administrator and run:
    echo   cd "d:\ML_prac\Minor_proj"
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install streamlit langchain langchain-community faiss-cpu transformers sentence-transformers pandas
    echo.
    pause
    exit /b 1
)

echo [2/3] Virtual environment found!
echo [3/3] Starting application...
echo.
echo Opening http://localhost:8502 in your browser...
echo.

"..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502

echo.
echo Application stopped.
pause
