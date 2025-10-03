# RAG Q&A Application - Simple Launcher
Write-Host "============================================" -ForegroundColor Green
Write-Host "RAG Q&A Application - Simple Launcher" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "Starting the RAG Q&A Application..." -ForegroundColor Yellow
Write-Host ""

# Try to use the existing virtual environment first
$venvPython = "..\..\..venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "Using virtual environment..." -ForegroundColor Green
    & $venvPython -m streamlit run app.py
}
else {
    Write-Host "Virtual environment not found. Using system Python..." -ForegroundColor Yellow
    
    # Try different Python commands
    $pythonCommands = @("python", "py", "python3")
    $success = $false
    
    foreach ($cmd in $pythonCommands) {
        try {
            Write-Host "Trying: $cmd" -ForegroundColor Cyan
            & $cmd -m streamlit run app.py
            $success = $true
            break
        }
        catch {
            Write-Host "Failed with $cmd, trying next..." -ForegroundColor Red
        }
    }
    
    if (-not $success) {
        Write-Host "All Python commands failed. Please ensure Python and Streamlit are installed." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
