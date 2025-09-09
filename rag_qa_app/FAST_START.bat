@echo off
echo ============================================
echo Fast Bhagavad Gita QA - Enhanced Version
echo ============================================
echo.

cd /d "%~dp0"

echo Starting the ENHANCED application...
echo Instant search + AI summaries!
echo.

"..\.venv\Scripts\python.exe" -m streamlit run fast_app.py --server.port 8503

echo.
echo Application stopped.
pause
