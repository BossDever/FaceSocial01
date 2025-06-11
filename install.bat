@echo off
echo 📦 Installing Face Recognition System
echo =====================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
python -m pip install -r requirements.txt

echo ✅ Installation completed!
echo.
echo To activate virtual environment: venv\Scripts\activate.bat
echo To run setup: setup.bat
echo To start system: start.bat
