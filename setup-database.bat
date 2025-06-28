@echo off
rem FaceSocial Database Setup Script for Windows
rem สำหรับติดตั้งและตั้งค่าฐานข้อมูล FaceSocial

setlocal enabledelayedexpansion

rem Database configuration
if not defined DB_HOST set DB_HOST=localhost
if not defined DB_PORT set DB_PORT=5432
if not defined DB_NAME set DB_NAME=facesocial
if not defined DB_USER set DB_USER=postgres
if not defined DB_PASSWORD set DB_PASSWORD=password

echo.
echo 🚀 FaceSocial Database Setup
echo ==================================
echo Host: %DB_HOST%:%DB_PORT%
echo Database: %DB_NAME%
echo User: %DB_USER%
echo.

rem Check if PostgreSQL is accessible
echo 🔍 Checking PostgreSQL connection...
set PGPASSWORD=%DB_PASSWORD%
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d postgres -c "SELECT version();" >nul 2>&1

if errorlevel 1 (
    echo ❌ Cannot connect to PostgreSQL
    echo Please check your database configuration and make sure PostgreSQL is running.
    pause
    exit /b 1
) else (
    echo ✅ PostgreSQL connection successful
)

rem Create database if it doesn't exist
echo 🏗️  Creating database if not exists...
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d postgres -c "CREATE DATABASE %DB_NAME%;" 2>nul
echo Database created or already exists

rem Check for required extensions
echo 🔌 Setting up required PostgreSQL extensions...
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" >nul 2>&1
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -c "CREATE EXTENSION IF NOT EXISTS \"vector\";" >nul 2>&1

rem Navigate to database directory
cd /d "%~dp0projec-final-fronend\database\init"
if errorlevel 1 (
    echo ❌ Cannot find database init directory
    pause
    exit /b 1
)

echo 📊 Setting up database schema...
echo.

rem Execute basic schema
if exist "01_schema.sql" (
    echo 📄 Executing: Basic schema (users, face_embeddings, sessions)
    psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -f "01_schema.sql"
    if errorlevel 1 (
        echo ❌ Failed to execute basic schema
        pause
        exit /b 1
    ) else (
        echo ✅ Success: Basic schema
    )
) else (
    echo ❌ 01_schema.sql not found
    pause
    exit /b 1
)

rem Execute social features
if exist "02_social_features.sql" (
    echo 📄 Executing: Social features (posts, chat, face tags)
    psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -f "02_social_features.sql"
    if errorlevel 1 (
        echo ❌ Failed to execute social features
        pause
        exit /b 1
    ) else (
        echo ✅ Success: Social features
    )
) else (
    echo ❌ 02_social_features.sql not found
    pause
    exit /b 1
)

rem Ask for sample data
if exist "03_sample_data.sql" (
    set /p "choice=Do you want to insert sample data? [y/N]: "
    if /i "!choice!"=="y" (
        echo 📄 Executing: Sample data for testing
        psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -f "03_sample_data.sql"
        if errorlevel 1 (
            echo ❌ Failed to insert sample data
        ) else (
            echo ✅ Success: Sample data
        )
    ) else (
        echo ℹ️  Skipping sample data
    )
)

rem Verify installation
echo 🔍 Verifying installation...
for /f "tokens=*" %%a in ('psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"') do set table_count=%%a

rem Remove leading/trailing spaces
set table_count=%table_count: =%

if %table_count% gtr 15 (
    echo ✅ Database setup completed successfully!
    echo 📊 Created %table_count% tables
) else (
    echo ❌ Database setup may have failed. Only %table_count% tables found.
    pause
    exit /b 1
)

echo.
echo 📋 Setup Summary
echo ==================================
echo ✅ Basic user management
echo ✅ Face recognition system
echo ✅ Posts with face tagging
echo ✅ Chat and messaging
echo ✅ Social connections
echo ✅ Notifications system
echo ✅ User profiles and preferences
echo.

echo 🔗 Database Connection Info
echo ==================================
echo Host: %DB_HOST%
echo Port: %DB_PORT%
echo Database: %DB_NAME%
echo User: %DB_USER%
echo.

echo 🚀 Next Steps
echo ==================================
echo 1. Update your .env file with database credentials
echo 2. Install required packages: npm install
echo 3. Start your application: npm run dev
echo 4. Visit http://localhost:3000 to test
echo.
echo 🎉 Happy coding with FaceSocial!

pause
