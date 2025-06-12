@echo off
echo 🚀 Face Recognition System Setup (Windows)
echo ===============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create directories
echo 📁 Creating directories...
if not exist "logs" mkdir "logs"
if not exist "output" mkdir "output"
if not exist "output\detection" mkdir "output\detection"
if not exist "output\recognition" mkdir "output\recognition"
if not exist "output\analysis" mkdir "output\analysis"
if not exist "temp" mkdir "temp"
if not exist "model" mkdir "model"
if not exist "model\face-detection" mkdir "model\face-detection"
if not exist "model\face-recognition" mkdir "model\face-recognition"

echo ✅ Directories created

REM Install requirements
echo 📦 Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install requirements
    
    pause
    exit /b 1
)

echo ✅ Requirements installed

REM Run Python setup
echo 🔧 Running Python setup...
python setup.py

if errorlevel 1 (
    echo ❌ Setup failed
    pause
    exit /b 1
)

echo ✅ Setup completed successfully!
echo.
echo 🚀 To start the system, run: start.bat
pause
