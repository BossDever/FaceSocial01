@echo off
echo 🎭 Face Recognition System Starter (Windows)
echo ============================================
echo.

REM Check if setup was run
if not exist "logs" (
    echo ❌ Setup not completed!
    echo 💡 Please run setup.bat first
    echo.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo 💡 Please install Python and run setup.bat
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment if available
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
    echo.
)

REM Parse command line arguments with new options
set RELOAD_ARG=--no-reload
set SKIP_CHECKS_ARG=
set LOG_LEVEL_ARG=INFO
set HOST_ARG=0.0.0.0
set PORT_ARG=8080
set MODE=production

:parse_args_loop
if "%1"=="" goto end_parse_args

if "%1"=="--reload" (
    set RELOAD_ARG=
    set MODE=development
    shift
    goto parse_args_loop
)
if "%1"=="--no-reload" (
    set RELOAD_ARG=--no-reload
    set MODE=production
    shift
    goto parse_args_loop
)
if "%1"=="--dev" (
    set RELOAD_ARG=
    set MODE=development
    set LOG_LEVEL_ARG=DEBUG
    shift
    goto parse_args_loop
)
if "%1"=="--prod" (
    set RELOAD_ARG=--no-reload
    set MODE=production
    set LOG_LEVEL_ARG=INFO
    shift
    goto parse_args_loop
)
if "%1"=="--skip-checks" (
    set SKIP_CHECKS_ARG=--skip-checks
    shift
    goto parse_args_loop
)
if "%1"=="--debug" (
    set LOG_LEVEL_ARG=DEBUG
    shift
    goto parse_args_loop
)
if "%1"=="--port" (
    set PORT_ARG=%2
    shift
    shift
    goto parse_args_loop
)
if "%1"=="--host" (
    set HOST_ARG=%2
    shift
    shift
    goto parse_args_loop
)
REM If argument is not recognized, shift and continue
shift
goto parse_args_loop

:end_parse_args

REM Quick health check
echo 🔍 Quick system check...
echo ✅ Mode: %MODE%
echo ✅ Reload: %RELOAD_ARG%
echo ✅ Log Level: %LOG_LEVEL_ARG%

python -c "import sys; print('✅ Python:', sys.version.split()[0])" 2>nul
if errorlevel 1 (
    echo ❌ Python check failed
    pause
    exit /b 1
)

python -c "import torch, cv2, fastapi; print('✅ Core packages available')" 2>nul
if errorlevel 1 (
    echo ❌ Core packages missing
    echo 💡 Run setup.bat to install requirements
    pause
    exit /b 1
)

REM Check model files (quick check)
set MODEL_COUNT=0
for %%F in ("model\face-detection\*.onnx") do set /a MODEL_COUNT+=1
for %%F in ("model\face-detection\*.pt") do set /a MODEL_COUNT+=1
for %%F in ("model\face-recognition\*.onnx") do set /a MODEL_COUNT+=1

if %MODEL_COUNT% GTR 0 (
    echo ✅ Model files detected (%MODEL_COUNT% types)
)
if NOT %MODEL_COUNT% GTR 0 (
    echo ⚠️ No model files found - limited functionality
    echo 💡 Place model files in model/ directory:
    echo    - model/face-detection/ (for YOLO models)
    echo    - model/face-recognition/ (for recognition models)
)

:start_system_loop
echo.
echo 🚀 Starting Face Recognition System...
echo ============================================
echo 🌐 Web Interface: http://%HOST_ARG%:%PORT_ARG%
echo 📚 API Documentation: http://%HOST_ARG%:%PORT_ARG%/docs
echo 📖 Alternative Docs: http://%HOST_ARG%:%PORT_ARG%/redoc
echo 🏥 Health Check: http://%HOST_ARG%:%PORT_ARG%/health
echo ============================================
echo.
echo 💡 Press Ctrl+C to stop the server
echo.

REM Start the system
if "%MODE%"=="development" (
    echo 🛠️ Running in DEVELOPMENT mode with auto-reload
    echo ⚠️ Use --prod for production mode without reload
) else (
    echo 🏭 Running in PRODUCTION mode (no auto-reload)
    echo 💡 Use --dev for development mode with reload
)
echo.

python start.py --host %HOST_ARG% --port %PORT_ARG% --log-level %LOG_LEVEL_ARG% %RELOAD_ARG% %SKIP_CHECKS_ARG%

REM Handle exit codes
if errorlevel 1 (
    echo.
    echo ❌ System failed to start or stopped with an error!
    echo.
    echo 🔍 Common issues and solutions:
    echo    1. Port %PORT_ARG% is already in use
    echo       💡 Try: start.bat --port %RANDOM:~-4% (random port)
    echo.
    echo    2. Missing dependencies  
    echo       💡 Try: setup.bat
    echo.
    echo    3. Model files missing or corrupted
    echo       💡 Place model files in model/ directory
    echo.
    echo    4. Reload loop issue (files changing constantly)
    echo       💡 Try: start.bat --prod (production mode)
    echo.
    echo    5. GPU/CUDA issues (if applicable)
    echo       💡 Check NVIDIA drivers and CUDA installation
    echo.
) else (
    echo.
    echo 👋 Face Recognition System stopped gracefully
    echo.
)

REM Ask if user wants to restart
:ask_restart
echo Available options:
echo   y - Restart with same settings
echo   d - Restart in development mode (with reload)
echo   p - Restart in production mode (no reload) 
echo   n - Exit
echo.
set /p RESTART_CHOICE=Choose option (y/d/p/n): 

if /i "%RESTART_CHOICE%"=="y" (
    echo.
    goto start_system_loop
)
if /i "%RESTART_CHOICE%"=="d" (
    set RELOAD_ARG=
    set MODE=development
    set LOG_LEVEL_ARG=DEBUG
    echo.
    echo 🛠️ Switching to development mode...
    goto start_system_loop
)
if /i "%RESTART_CHOICE%"=="p" (
    set RELOAD_ARG=--no-reload
    set MODE=production  
    set LOG_LEVEL_ARG=INFO
    echo.
    echo 🏭 Switching to production mode...
    goto start_system_loop
)
if /i "%RESTART_CHOICE%"=="n" (
    goto end_script
)
echo Invalid choice. Please enter 'y', 'd', 'p', or 'n'.
goto ask_restart

:end_script
echo 👍 Goodbye!
pause
exit /b 0
