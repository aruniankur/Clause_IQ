#!/bin/bash

# Clause_IQ Docker Startup Script for Mac/Linux
# This script builds and starts the Clause_IQ application using Docker

set -e

echo "=========================================="
echo "  Clause_IQ Docker Startup Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed."
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Error: Docker Compose is not available."
    echo "Please install Docker Compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please update it with your configuration."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p standard_tempate_default

# Check if we should build or just start
if [ "$1" == "--rebuild" ]; then
    echo "🔨 Rebuilding Docker image..."
    docker-compose down
    docker-compose build --no-cache
elif [ "$1" == "--clean" ]; then
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down -v
    docker system prune -f
    echo "✅ Cleanup complete!"
    exit 0
fi

# Start the application
echo "🚀 Starting Clause_IQ application..."
docker-compose up -d

# Wait for the application to be ready
echo "⏳ Waiting for application to be ready..."
sleep 5

# Check if container is running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Clause_IQ is now running!"
    echo "🌐 Access the application at: http://localhost:5000"
    echo ""
    echo "Useful commands:"
    echo "  • View logs:        docker-compose logs -f"
    echo "  • Stop application: docker-compose down"
    echo "  • Restart:          docker-compose restart"
    echo "  • Rebuild:          ./start-docker.sh --rebuild"
    echo "  • Clean up:         ./start-docker.sh --clean"
    echo ""
else
    echo "❌ Error: Container failed to start. Check logs with:"
    echo "   docker-compose logs"
    exit 1
fi
