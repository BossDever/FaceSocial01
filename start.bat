@echo off
echo 🎭 Face Recognition System Starter (Windows)
echo ============================================

REM Check if setup was run
if not exist "logs" (
    echo ❌ Please run setup.bat first
    pause
    exit /b 1
)

REM Start system
echo 🚀 Starting Face Recognition System...
echo Web Interface: http://localhost:8080
echo API Docs: http://localhost:8080/docs
echo.

python start.py

if errorlevel 1 (
    echo ❌ System failed to start
    pause
    exit /b 1
)
