#!/bin/bash
# Docker Build and Run Script for Face Recognition System

set -e

echo "🚀 Face Recognition System - Docker Build & Run"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is running"
}

# Check NVIDIA Docker support
check_nvidia_docker() {
    print_status "Checking NVIDIA Docker support..."
    if ! command -v nvidia-docker &> /dev/null && ! docker info | grep -q nvidia; then
        print_warning "NVIDIA Docker support not detected. GPU acceleration may not work."
        print_warning "Install nvidia-docker2 for GPU support."
    else
        print_success "NVIDIA Docker support detected"
    fi
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    print_status "This may take 10-20 minutes for the first build..."
    
    docker build -t face-recognition-system:latest . \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Run with docker-compose
run_compose() {
    print_status "Starting services with docker-compose..."
    
    # Create necessary directories
    mkdir -p output/{detection,recognition,analysis} logs
    
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_success "Services started successfully"
        print_status "API will be available at: http://localhost:8000"
        print_status "Health check: http://localhost:8000/health"
        print_status "API docs: http://localhost:8000/docs"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Run standalone container
run_standalone() {
    print_status "Running standalone container..."
    
    docker run -d \
        --name face-recognition-container \
        --gpus all \
        -p 8000:8000 \
        -v "$(pwd)/model:/app/model:ro" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/test_images:/app/test_images:ro" \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e PYTHONUNBUFFERED=1 \
        --restart unless-stopped \
        face-recognition-system:latest
    
    if [ $? -eq 0 ]; then
        print_success "Container started successfully"
        print_status "Container name: face-recognition-container"
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Show logs
show_logs() {
    print_status "Showing container logs..."
    docker-compose logs -f face-recognition-api
}

# Stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v
    docker image prune -f
    docker container prune -f
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build         Build Docker image"
    echo "  run           Run with docker-compose (recommended)"
    echo "  standalone    Run standalone container"
    echo "  logs          Show container logs"
    echo "  stop          Stop all services"
    echo "  restart       Restart services"
    echo "  cleanup       Clean up Docker resources"
    echo "  status        Show service status"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build       # Build the image"
    echo "  $0 run         # Start services"
    echo "  $0 logs        # View logs"
    echo "  $0 stop        # Stop services"
}

# Show status
show_status() {
    print_status "Service status:"
    docker-compose ps
    echo ""
    print_status "Docker images:"
    docker images | grep face-recognition
}

# Restart services
restart_services() {
    print_status "Restarting services..."
    docker-compose restart
    print_success "Services restarted"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        check_nvidia_docker
        build_image
        ;;
    "run")
        check_docker
        check_nvidia_docker
        run_compose
        ;;
    "standalone")
        check_docker
        check_nvidia_docker
        run_standalone
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        show_status
        ;;
    "help"|*)
        show_help
        ;;
esac
