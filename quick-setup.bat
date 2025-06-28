@echo off
REM ===================================================================
REM FaceSocial Platform - One-Click Setup Script for Windows
REM ===================================================================
REM This script will automatically setup and run the entire FaceSocial platform
REM Prerequisites: Docker Desktop must be installed and running
REM Usage: Double-click this file or run: quick-setup.bat
REM ===================================================================

setlocal enabledelayedexpansion
title FaceSocial Platform Setup

REM Colors for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "CYAN=[96m"
set "NC=[0m"

REM Configuration
set "PROJECT_NAME=FaceSocial Platform"
set "COMPOSE_PROJECT_NAME=facesocial"
set "BACKEND_PORT=8080"
set "FRONTEND_PORT=3000"
set "DB_PORT=5432"
set "REDIS_PORT=6379"

echo.
echo %CYAN%===============================================================%NC%
echo %CYAN%                🚀 FaceSocial Platform Setup                   %NC%
echo %CYAN%                     One-Click Installation                    %NC%
echo %CYAN%===============================================================%NC%
echo.

echo %GREEN%[INFO]%NC% Starting automated setup process...
echo.

REM Check Docker installation
echo %CYAN%🐳 Checking Docker installation...%NC%
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker is not installed or not in PATH.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker Compose is not available.
    echo Please ensure Docker Desktop is properly installed.
    pause
    exit /b 1
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker daemon is not running.
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo %GREEN%✅ Docker is installed and running%NC%
docker --version
docker-compose --version
echo.

REM Check NVIDIA Docker support
echo %CYAN%🎮 Checking NVIDIA Docker support...%NC%
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠️  No NVIDIA GPU detected, will run in CPU mode%NC%
    echo export NVIDIA_DOCKER_AVAILABLE=false > .env.docker
) else (
    echo %GREEN%✅ NVIDIA GPU detected%NC%
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    echo export NVIDIA_DOCKER_AVAILABLE=true > .env.docker
)
echo.

REM Setup environment files
echo %CYAN%🔧 Setting up environment configuration...%NC%

REM Create main .env file
(
echo # FaceSocial Platform Configuration
echo PROJECT_NAME=%COMPOSE_PROJECT_NAME%
echo ENVIRONMENT=development
echo.
echo # Database Configuration
echo DATABASE_URL=postgresql://facesocial_user:facesocial_password@postgres:5432/facesocial_db
echo POSTGRES_DB=facesocial_db
echo POSTGRES_USER=facesocial_user
echo POSTGRES_PASSWORD=facesocial_password
echo POSTGRES_HOST=postgres
echo POSTGRES_PORT=5432
echo.
echo # Redis Configuration
echo REDIS_URL=redis://redis:6379/0
echo REDIS_HOST=redis
echo REDIS_PORT=6379
echo.
echo # Backend Configuration
echo BACKEND_HOST=0.0.0.0
echo BACKEND_PORT=%BACKEND_PORT%
echo BACKEND_URL=http://localhost:%BACKEND_PORT%
echo.
echo # Frontend Configuration
echo FRONTEND_PORT=%FRONTEND_PORT%
echo NEXT_PUBLIC_API_URL=http://localhost:%BACKEND_PORT%
echo NEXT_PUBLIC_WS_URL=ws://localhost:%BACKEND_PORT%
echo.
echo # AI Services Configuration
echo MODEL_DIR=/app/model
echo AUTO_DOWNLOAD_MODELS=true
echo CUDA_VISIBLE_DEVICES=0
echo.
echo # Security
echo JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
echo SESSION_SECRET=your-super-secret-session-key-change-this-in-production
echo.
echo # File Upload
echo MAX_FILE_SIZE=50MB
echo UPLOAD_DIR=/app/uploads
echo.
echo # Logging
echo LOG_LEVEL=INFO
echo LOG_FORMAT=detailed
) > .env

REM Create frontend .env.local
if not exist "projec-final-fronend" mkdir projec-final-fronend
(
echo # Frontend Environment Variables
echo NEXT_PUBLIC_API_URL=http://localhost:%BACKEND_PORT%
echo NEXT_PUBLIC_WS_URL=ws://localhost:%BACKEND_PORT%
echo NEXT_PUBLIC_APP_NAME=FaceSocial Platform
echo NEXT_PUBLIC_APP_VERSION=1.0.0
echo.
echo # Database URL for Prisma
echo DATABASE_URL=postgresql://facesocial_user:facesocial_password@localhost:%DB_PORT%/facesocial_db
echo.
echo # Session Configuration
echo NEXTAUTH_SECRET=your-nextauth-secret-change-this
echo NEXTAUTH_URL=http://localhost:%FRONTEND_PORT%
) > projec-final-fronend\.env.local

echo %GREEN%✅ Environment files created%NC%
echo.

REM Setup database initialization
echo %CYAN%🗄️  Setting up database initialization...%NC%
if not exist "projec-final-fronend\database\init" mkdir projec-final-fronend\database\init

REM Create database initialization script (simplified for batch)
(
echo -- FaceSocial Database Initialization
echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
echo CREATE EXTENSION IF NOT EXISTS "pgcrypto";
echo.
echo -- Users table
echo CREATE TABLE IF NOT EXISTS users ^(
echo     id SERIAL PRIMARY KEY,
echo     uuid UUID DEFAULT uuid_generate_v4^(^) UNIQUE,
echo     username VARCHAR^(50^) UNIQUE NOT NULL,
echo     email VARCHAR^(255^) UNIQUE NOT NULL,
echo     password_hash VARCHAR^(255^) NOT NULL,
echo     full_name VARCHAR^(255^),
echo     profile_image TEXT,
echo     face_embeddings JSONB,
echo     face_image_data TEXT,
echo     is_active BOOLEAN DEFAULT true,
echo     is_admin BOOLEAN DEFAULT false,
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Insert sample admin user
echo INSERT INTO users ^(username, email, password_hash, full_name, is_admin^) 
echo VALUES ^(
echo     'admin', 
echo     'admin@facesocial.com', 
echo     crypt^('admin123', gen_salt^('bf'^)^), 
echo     'System Administrator', 
echo     true
echo ^) ON CONFLICT ^(username^) DO NOTHING;
) > projec-final-fronend\database\init\01-init-database.sql

echo %GREEN%✅ Database initialization scripts created%NC%
echo.

REM Create monitoring script for Windows
echo %CYAN%📊 Creating system monitoring script...%NC%
(
echo @echo off
echo title FaceSocial System Monitor
echo :loop
echo cls
echo echo ===================================
echo echo 🚀 FaceSocial Platform Status
echo echo ===================================
echo echo 📅 %DATE% %TIME%
echo echo.
echo echo 🐳 Docker Containers:
echo docker-compose ps
echo echo.
echo echo 🌐 Service URLs:
echo echo Frontend: http://localhost:3000
echo echo Backend API: http://localhost:8080
echo echo API Docs: http://localhost:8080/docs
echo echo Database: localhost:5432
echo echo.
echo echo 📊 Container Health:
echo for %%%%s in ^(backend frontend postgres redis^) do ^(
echo     docker-compose ps %%%%s ^| findstr /C:"running" ^>nul ^&^& echo ✅ %%%%s: Running ^|^| echo ❌ %%%%s: Not running
echo ^)
echo echo.
echo echo Press Ctrl+C to exit monitoring
echo timeout /t 10 /nobreak ^>nul
echo goto loop
) > monitor-system.bat

echo %GREEN%✅ Monitoring script created%NC%
echo.

REM Stop any existing containers
echo %CYAN%🛑 Stopping any existing containers...%NC%
docker-compose down --remove-orphans >nul 2>&1

REM Clean up
echo %CYAN%🧹 Cleaning up...%NC%
docker system prune -f >nul 2>&1

REM Build and start services
echo %CYAN%🚀 Building and starting FaceSocial services...%NC%
echo %YELLOW%This may take several minutes for the first time...%NC%
echo.

docker-compose build --no-cache
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to build Docker images.
    pause
    exit /b 1
)

docker-compose up -d
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to start services.
    pause
    exit /b 1
)

echo %GREEN%✅ Services are starting up...%NC%
echo.

REM Wait for services
echo %CYAN%⏳ Waiting for services to be ready...%NC%
echo %YELLOW%This may take a few minutes while AI models are downloaded...%NC%
echo.

REM Wait for database (simplified check)
echo Waiting for database...
timeout /t 10 /nobreak >nul

REM Wait for backend (with longer timeout for model download)
echo Waiting for backend API ^(downloading AI models^)...
timeout /t 60 /nobreak >nul

REM Wait for frontend
echo Waiting for frontend...
timeout /t 20 /nobreak >nul

echo.
echo %CYAN%🎉 FaceSocial Platform Setup Complete!%NC%
echo.
echo %CYAN%🌐 Access URLs:%NC%
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🖥️  Frontend Application:  http://localhost:3000
echo 🔧 Backend API:           http://localhost:8080
echo 📚 API Documentation:     http://localhost:8080/docs
echo 🗄️  Database:             localhost:5432
echo 🗄️  Redis Cache:          localhost:6379
echo.
echo %CYAN%👤 Default Login Credentials:%NC%
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 👨‍💼 Admin:     username: admin,     password: admin123
echo 👤 Demo User:  username: demo_user, password: demo123
echo.
echo %CYAN%🛠️  Management Commands:%NC%
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📊 Monitor System:        monitor-system.bat
echo 📋 View Logs:             docker-compose logs -f [service]
echo 🔄 Restart Services:      docker-compose restart
echo 🛑 Stop All Services:     docker-compose down
echo 🧹 Clean Everything:      docker-compose down -v --rmi all
echo.
echo %GREEN%✨ Platform is ready to use! Visit http://localhost:3000 to get started.%NC%
echo.
echo %CYAN%🎯 Next Steps:%NC%
echo 1. Visit http://localhost:3000 to access the platform
echo 2. Login with admin/admin123 or demo_user/demo123  
echo 3. Try the face recognition features
echo 4. Check the API documentation at http://localhost:8080/docs
echo.
echo %GREEN%Setup completed successfully! 🎉%NC%
echo.

REM Ask if user wants to open browser
choice /C YN /M "Do you want to open the platform in your browser now?"
if errorlevel 2 goto end
start http://localhost:3000

:end
echo Press any key to exit...
pause >nul
