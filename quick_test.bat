@echo off
REM Quick API Test Runner
REM For rapid development testing

echo ============================
echo Quick API Test
echo ============================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check API server
python -c "import requests; requests.get('http://localhost:8080/health', timeout=3)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: API server not accessible
    echo Please start the server first
    pause
    exit /b 1
)

REM Install requirements
pip install requests >nul 2>&1

echo Running quick test...
echo.

REM Run quick test
python quick_api_test.py

echo.
echo Test completed!
pause
