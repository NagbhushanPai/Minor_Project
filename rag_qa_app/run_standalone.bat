@echo off
echo ============================================
echo RAG Q&A Application - Standalone Runner
echo ============================================
echo.

cd /d "%~dp0"

echo Running standalone Python launcher...
python run_app_standalone.py

if %errorlevel% neq 0 (
    echo.
    echo ============================================
    echo Alternative: Try with different Python commands
    echo ============================================
    echo Trying with 'py' command...
    py run_app_standalone.py
    
    if %errorlevel% neq 0 (
        echo.
        echo Trying with 'python3' command...
        python3 run_app_standalone.py
    )
)

echo.
echo Press any key to exit...
pause >nul
