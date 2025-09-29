@echo off
REM Clause_IQ Docker Startup Script for Windows
REM This script builds and starts the Clause_IQ application using Docker

echo ==========================================
echo   Clause_IQ Docker Startup Script
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker is not installed.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo .env file created. Please update it with your configuration.
)

REM Create necessary directories
echo Creating necessary directories...
if not exist uploads mkdir uploads
if not exist standard_tempate_default mkdir standard_tempate_default

REM Check command line arguments
if "%1"=="--rebuild" (
    echo Rebuilding Docker image...
    docker-compose down
    docker-compose build --no-cache
) else if "%1"=="--clean" (
    echo Cleaning up Docker resources...
    docker-compose down -v
    docker system prune -f
    echo Cleanup complete!
    pause
    exit /b 0
)

REM Start the application
echo Starting Clause_IQ application...
docker-compose up -d

REM Wait for the application to be ready
echo Waiting for application to be ready...
timeout /t 5 /nobreak >nul

REM Check if container is running
docker-compose ps | findstr /C:"Up" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Clause_IQ is now running!
    echo Access the application at: http://localhost:5000
    echo.
    echo Useful commands:
    echo   * View logs:        docker-compose logs -f
    echo   * Stop application: docker-compose down
    echo   * Restart:          docker-compose restart
    echo   * Rebuild:          start-docker.bat --rebuild
    echo   * Clean up:         start-docker.bat --clean
    echo.
) else (
    echo Error: Container failed to start. Check logs with:
    echo    docker-compose logs
    pause
    exit /b 1
)

pause
