@echo off
REM Docker Build and Run Script for Face Recognition System (Windows)

setlocal enabledelayedexpansion

echo 🚀 Face Recognition System - Docker Build ^& Run
echo ================================================

REM Check if Docker is running
echo [INFO] Checking Docker installation...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is running

REM Check command line argument
if "%1"=="build" goto build
if "%1"=="run" goto run
if "%1"=="standalone" goto standalone
if "%1"=="logs" goto logs
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="cleanup" goto cleanup
if "%1"=="status" goto status
goto help

:build
echo [INFO] Building Docker image...
echo [INFO] This may take 10-20 minutes for the first build...
docker build -t face-recognition-system:latest .
if %errorlevel% equ 0 (
    echo [SUCCESS] Docker image built successfully
) else (
    echo [ERROR] Failed to build Docker image
    pause
    exit /b 1
)
goto end

:run
echo [INFO] Starting services with docker-compose...
REM Create necessary directories
if not exist "output\detection" mkdir output\detection
if not exist "output\recognition" mkdir output\recognition  
if not exist "output\analysis" mkdir output\analysis
if not exist "logs" mkdir logs

docker-compose up -d
if %errorlevel% equ 0 (
    echo [SUCCESS] Services started successfully
    echo [INFO] API will be available at: http://localhost:8000
    echo [INFO] Health check: http://localhost:8000/health
    echo [INFO] API docs: http://localhost:8000/docs
) else (
    echo [ERROR] Failed to start services
    pause
    exit /b 1
)
goto end

:standalone
echo [INFO] Running standalone container...
docker run -d --name face-recognition-container --gpus all -p 8000:8000 -v "%cd%\model:/app/model:ro" -v "%cd%\output:/app/output" -v "%cd%\logs:/app/logs" -v "%cd%\test_images:/app/test_images:ro" -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 --restart unless-stopped face-recognition-system:latest
if %errorlevel% equ 0 (
    echo [SUCCESS] Container started successfully
    echo [INFO] Container name: face-recognition-container
) else (
    echo [ERROR] Failed to start container
    pause
    exit /b 1
)
goto end

:logs
echo [INFO] Showing container logs...
docker-compose logs -f face-recognition-api
goto end

:stop
echo [INFO] Stopping services...
docker-compose down
echo [SUCCESS] Services stopped
goto end

:restart
echo [INFO] Restarting services...
docker-compose restart
echo [SUCCESS] Services restarted
goto end

:cleanup
echo [INFO] Cleaning up Docker resources...
docker-compose down -v
docker image prune -f
docker container prune -f
echo [SUCCESS] Cleanup completed
goto end

:status
echo [INFO] Service status:
docker-compose ps
echo.
echo [INFO] Docker images:
docker images | findstr face-recognition
goto end

:help
echo Usage: %0 [OPTION]
echo.
echo Options:
echo   build         Build Docker image
echo   run           Run with docker-compose (recommended)
echo   standalone    Run standalone container
echo   logs          Show container logs
echo   stop          Stop all services
echo   restart       Restart services
echo   cleanup       Clean up Docker resources
echo   status        Show service status
echo   help          Show this help message
echo.
echo Examples:
echo   %0 build       # Build the image
echo   %0 run         # Start services
echo   %0 logs        # View logs
echo   %0 stop        # Stop services

:end
if not "%1"=="logs" pause
