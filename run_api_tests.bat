@echo off
REM Comprehensive API Testing Script for Face Recognition System
REM Created: June 14, 2025

echo ================================
echo Face Recognition API Test Suite
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo Python is available
echo.

REM Check if test images directory exists
if not exist "test_images" (
    echo ERROR: test_images directory not found
    echo Please ensure test_images directory exists with test images
    pause
    exit /b 1
)

echo Test images directory found
echo.

REM Check if API server is running
echo Checking API server availability...
python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: API server is not running or not accessible
    echo Please start the API server first by running:
    echo   python src/main.py
    echo   or
    echo   start.bat
    pause
    exit /b 1
)

echo API server is accessible
echo.

REM Install required packages
echo Installing required packages...
pip install requests

echo.
echo ================================
echo Starting API Tests...
echo ================================
echo.

REM Run the API tests
python simple_api_tester.py

echo.
echo ================================
echo Test completed!
echo ================================
echo.

REM Show output directory
if exist "output\api_test" (
    echo Test results saved to: output\api_test\
    echo Log file: api_test_log.txt
) else (
    echo Warning: Output directory not found
)

echo.
pause
