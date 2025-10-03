#!/usr/bin/env python3
"""
Standalone runner for RAG Q&A Application
This script automatically handles Python environment setup and runs the application
"""

import sys
import subprocess
import os
from pathlib import Path

def find_python_executable():
    """Find the best Python executable to use"""
    # Get the current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Check for virtual environment in various locations
    venv_locations = [
        script_dir / "rag_env" / "Scripts" / "python.exe",  # Windows venv in project
        script_dir.parent / ".venv" / "Scripts" / "python.exe",  # Windows venv in parent
        script_dir / "venv" / "Scripts" / "python.exe",  # Alternative Windows venv
        script_dir / "rag_env" / "bin" / "python",  # Unix venv in project
        script_dir.parent / ".venv" / "bin" / "python",  # Unix venv in parent
    ]
    
    # Check virtual environments first
    for venv_python in venv_locations:
        if venv_python.exists():
            print(f"✅ Found virtual environment Python: {venv_python}")
            return str(venv_python)
    
    # Fall back to system Python
    system_pythons = ["python", "python3", "py"]
    for python_cmd in system_pythons:
        try:
            result = subprocess.run([python_cmd, "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Found system Python: {python_cmd} ({result.stdout.strip()})")
            return python_cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    raise RuntimeError("❌ No suitable Python executable found!")

def check_required_packages(python_exe):
    """Check if required packages are installed"""
    required_packages = [
        "streamlit", "langchain", "langchain-community", "faiss-cpu", 
        "sentence-transformers", "transformers", "pandas"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            subprocess.run([python_exe, "-c", f"import {package.replace('-', '_')}"], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError:
            missing_packages.append(package)
    
    return missing_packages

def install_packages(python_exe, packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.run([python_exe, "-m", "pip", "install"] + packages, check=True)
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def run_streamlit_app(python_exe):
    """Run the Streamlit application"""
    script_dir = Path(__file__).parent.absolute()
    app_path = script_dir / "app.py"
    
    if not app_path.exists():
        raise FileNotFoundError(f"❌ Application file not found: {app_path}")
    
    print(f"🚀 Starting Streamlit application...")
    print(f"📂 Working directory: {script_dir}")
    print(f"🐍 Using Python: {python_exe}")
    print(f"📄 Running: {app_path}")
    print("=" * 50)
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run Streamlit
    subprocess.run([python_exe, "-m", "streamlit", "run", str(app_path)])

def main():
    """Main function"""
    print("🤖 RAG Q&A Application Standalone Runner")
    print("=" * 50)
    
    try:
        # Find Python executable
        python_exe = find_python_executable()
        
        # Check required packages
        missing_packages = check_required_packages(python_exe)
        
        if missing_packages:
            print(f"⚠️  Missing packages detected: {', '.join(missing_packages)}")
            user_input = input("Install missing packages automatically? (y/n): ").lower().strip()
            
            if user_input in ['y', 'yes']:
                if not install_packages(python_exe, missing_packages):
                    print("❌ Failed to install required packages. Please install manually.")
                    return 1
            else:
                print("❌ Cannot run without required packages. Please install them manually:")
                print(f"   {python_exe} -m pip install {' '.join(missing_packages)}")
                return 1
        else:
            print("✅ All required packages are installed!")
        
        # Run the application
        run_streamlit_app(python_exe)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
