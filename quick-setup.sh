#!/bin/bash
# ===================================================================
# FaceSocial Platform - One-Click Setup Script
# ===================================================================
# This script will automatically setup and run the entire FaceSocial platform
# Prerequisites: Docker and Docker Compose must be installed
# Usage: chmod +x quick-setup.sh && ./quick-setup.sh
# ===================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="FaceSocial Platform"
COMPOSE_PROJECT_NAME="facesocial"
BACKEND_PORT=8080
FRONTEND_PORT=3000
DB_PORT=5432
REDIS_PORT=6379

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker installation
check_docker() {
    print_header "🐳 Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Download from: https://www.docker.com/get-started"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "✅ Docker is installed and running"
    docker --version
    docker-compose --version
}

# Function to check NVIDIA Docker (for GPU support)
check_nvidia_docker() {
    print_header "🎮 Checking NVIDIA Docker support..."
    
    if command_exists nvidia-smi; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || true
        
        if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
            print_success "✅ NVIDIA Docker support is available"
            echo "export NVIDIA_DOCKER_AVAILABLE=true" > .env.docker
        else
            print_warning "⚠️  NVIDIA Docker not properly configured, will run in CPU mode"
            echo "export NVIDIA_DOCKER_AVAILABLE=false" > .env.docker
        fi
    else
        print_warning "⚠️  No NVIDIA GPU detected, will run in CPU mode"
        echo "export NVIDIA_DOCKER_AVAILABLE=false" > .env.docker
    fi
}

# Function to setup environment files
setup_environment() {
    print_header "🔧 Setting up environment configuration..."
    
    # Create main .env file
    cat > .env << EOF
# FaceSocial Platform Configuration
PROJECT_NAME=${COMPOSE_PROJECT_NAME}
ENVIRONMENT=development

# Database Configuration
DATABASE_URL=postgresql://facesocial_user:facesocial_password@postgres:5432/facesocial_db
POSTGRES_DB=facesocial_db
POSTGRES_USER=facesocial_user
POSTGRES_PASSWORD=facesocial_password
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_HOST=redis
REDIS_PORT=6379

# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=${BACKEND_PORT}
BACKEND_URL=http://localhost:${BACKEND_PORT}

# Frontend Configuration
FRONTEND_PORT=${FRONTEND_PORT}
NEXT_PUBLIC_API_URL=http://localhost:${BACKEND_PORT}
NEXT_PUBLIC_WS_URL=ws://localhost:${BACKEND_PORT}

# AI Services Configuration
MODEL_DIR=/app/model
AUTO_DOWNLOAD_MODELS=true
CUDA_VISIBLE_DEVICES=0

# Security
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
SESSION_SECRET=your-super-secret-session-key-change-this-in-production

# File Upload
MAX_FILE_SIZE=50MB
UPLOAD_DIR=/app/uploads

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed
EOF

    # Create frontend .env.local
    mkdir -p projec-final-fronend
    cat > projec-final-fronend/.env.local << EOF
# Frontend Environment Variables
NEXT_PUBLIC_API_URL=http://localhost:${BACKEND_PORT}
NEXT_PUBLIC_WS_URL=ws://localhost:${BACKEND_PORT}
NEXT_PUBLIC_APP_NAME=FaceSocial Platform
NEXT_PUBLIC_APP_VERSION=1.0.0

# Database URL for Prisma
DATABASE_URL=postgresql://facesocial_user:facesocial_password@localhost:${DB_PORT}/facesocial_db

# Session Configuration
NEXTAUTH_SECRET=your-nextauth-secret-change-this
NEXTAUTH_URL=http://localhost:${FRONTEND_PORT}
EOF

    print_success "✅ Environment files created"
}

# Function to create complete docker-compose.yml
create_docker_compose() {
    print_header "🐙 Creating Docker Compose configuration..."
    
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: facesocial_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./projec-final-fronend/database/init:/docker-entrypoint-initdb.d
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - facesocial_network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: facesocial_redis
    restart: unless-stopped
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - facesocial_network

  # Backend AI Services
  backend:
    build: 
      context: .
      dockerfile: Dockerfile
    image: facesocial-backend:latest
    container_name: facesocial_backend
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
      - PYTHONUNBUFFERED=1
      - MODEL_DIR=/app/model
      - AUTO_DOWNLOAD_MODELS=true
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    ports:
      - "${BACKEND_PORT:-8080}:8080"
    volumes:
      - model_data:/app/model
      - upload_data:/app/uploads
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
    networks:
      - facesocial_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  # Frontend Application
  frontend:
    build:
      context: ./projec-final-fronend
      dockerfile: Dockerfile
    image: facesocial-frontend:latest
    container_name: facesocial_frontend
    restart: unless-stopped
    depends_on:
      backend:
        condition: service_healthy
      postgres:
        condition: service_healthy
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:${BACKEND_PORT:-8080}
      - DATABASE_URL=${DATABASE_URL}
      - NEXTAUTH_SECRET=${JWT_SECRET}
      - NEXTAUTH_URL=http://localhost:${FRONTEND_PORT:-3000}
    ports:
      - "${FRONTEND_PORT:-3000}:3000"
    volumes:
      - ./projec-final-fronend/.env.local:/app/.env.local:ro
    networks:
      - facesocial_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  model_data:
    driver: local
  upload_data:
    driver: local

networks:
  facesocial_network:
    driver: bridge
EOF

    # Create GPU-enabled docker-compose override if NVIDIA Docker is available
    if [ -f .env.docker ] && grep -q "NVIDIA_DOCKER_AVAILABLE=true" .env.docker; then
        cat > docker-compose.gpu.yml << 'EOF'
version: '3.8'

services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    runtime: nvidia
EOF
        print_status "Created GPU-enabled Docker Compose override"
    fi

    print_success "✅ Docker Compose configuration created"
}

# Function to setup database initialization
setup_database_init() {
    print_header "🗄️  Setting up database initialization..."
    
    mkdir -p projec-final-fronend/database/init
    
    # Create comprehensive database initialization script
    cat > projec-final-fronend/database/init/01-init-database.sql << 'EOF'
-- FaceSocial Database Initialization
-- This script creates the complete database schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    profile_image TEXT,
    face_embeddings JSONB,
    face_image_data TEXT,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Posts table
CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    content TEXT,
    image_url TEXT,
    face_tags JSONB DEFAULT '[]',
    likes_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Comments table
CREATE TABLE IF NOT EXISTS comments (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Likes table
CREATE TABLE IF NOT EXISTS likes (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, user_id)
);

-- Friendships table
CREATE TABLE IF NOT EXISTS friendships (
    id SERIAL PRIMARY KEY,
    requester_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    addressee_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'declined', 'blocked')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(requester_id, addressee_id)
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    sender_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    receiver_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    is_read BOOLEAN DEFAULT false,
    message_type VARCHAR(20) DEFAULT 'text' CHECK (message_type IN ('text', 'image', 'file')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    data JSONB DEFAULT '{}',
    is_read BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face recognition data table
CREATE TABLE IF NOT EXISTS face_data (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    face_encoding JSONB NOT NULL,
    image_path TEXT,
    quality_score FLOAT,
    model_version VARCHAR(50) DEFAULT 'facenet',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System activities table
CREATE TABLE IF NOT EXISTS activities (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id INTEGER,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_likes_post_id ON likes(post_id);
CREATE INDEX IF NOT EXISTS idx_friendships_users ON friendships(requester_id, addressee_id);
CREATE INDEX IF NOT EXISTS idx_messages_participants ON messages(sender_id, receiver_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_face_data_user_id ON face_data(user_id);
CREATE INDEX IF NOT EXISTS idx_activities_user_id ON activities(user_id);
CREATE INDEX IF NOT EXISTS idx_activities_created_at ON activities(created_at DESC);

-- Insert sample admin user
INSERT INTO users (username, email, password_hash, full_name, is_admin) 
VALUES (
    'admin', 
    'admin@facesocial.com', 
    crypt('admin123', gen_salt('bf')), 
    'System Administrator', 
    true
) ON CONFLICT (username) DO NOTHING;

-- Insert sample regular user
INSERT INTO users (username, email, password_hash, full_name) 
VALUES (
    'demo_user', 
    'demo@facesocial.com', 
    crypt('demo123', gen_salt('bf')), 
    'Demo User'
) ON CONFLICT (username) DO NOTHING;

-- Create trigger for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_posts_updated_at ON posts;
CREATE TRIGGER update_posts_updated_at BEFORE UPDATE ON posts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_friendships_updated_at ON friendships;
CREATE TRIGGER update_friendships_updated_at BEFORE UPDATE ON friendships FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO facesocial_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO facesocial_user;
EOF

    print_success "✅ Database initialization scripts created"
}

# Function to create startup monitoring script
create_monitoring_script() {
    print_header "📊 Creating system monitoring script..."
    
    cat > monitor-system.sh << 'EOF'
#!/bin/bash
# FaceSocial System Monitor

while true; do
    clear
    echo "==================================="
    echo "🚀 FaceSocial Platform Status"
    echo "==================================="
    echo "📅 $(date)"
    echo ""
    
    echo "🐳 Docker Containers:"
    docker-compose ps
    echo ""
    
    echo "💾 System Resources:"
    echo "Memory Usage: $(free -h | awk '/^Mem:/ { print $3"/"$2 }')"
    echo "Disk Usage: $(df -h . | awk 'NR==2 { print $3"/"$2" ("$5")" }')"
    echo ""
    
    echo "🌐 Service URLs:"
    echo "Frontend: http://localhost:3000"
    echo "Backend API: http://localhost:8080"
    echo "API Docs: http://localhost:8080/docs"
    echo "Database: localhost:5432"
    echo ""
    
    echo "📊 Container Health:"
    for service in backend frontend postgres redis; do
        health=$(docker-compose ps $service | grep -E "(healthy|running)" || echo "unhealthy")
        if [[ $health == *"healthy"* ]] || [[ $health == *"running"* ]]; then
            echo "✅ $service: Running"
        else
            echo "❌ $service: Not running"
        fi
    done
    
    echo ""
    echo "Press Ctrl+C to exit monitoring"
    sleep 10
done
EOF

    chmod +x monitor-system.sh
    print_success "✅ Monitoring script created"
}

# Function to build and start services
start_services() {
    print_header "🚀 Building and starting FaceSocial services..."
    
    # Stop any existing containers
    print_status "Stopping any existing containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Clean up any dangling containers
    print_status "Cleaning up..."
    docker system prune -f >/dev/null 2>&1 || true
    
    # Build and start services
    print_status "Building Docker images (this may take several minutes)..."
    
    if [ -f docker-compose.gpu.yml ] && [ -f .env.docker ] && grep -q "NVIDIA_DOCKER_AVAILABLE=true" .env.docker; then
        print_status "Starting with GPU support..."
        docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache
        docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
    else
        print_status "Starting in CPU mode..."
        docker-compose build --no-cache
        docker-compose up -d
    fi
    
    print_success "✅ Services are starting up..."
}

# Function to wait for services to be ready
wait_for_services() {
    print_header "⏳ Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for database..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec -T postgres pg_isready -U facesocial_user -d facesocial_db >/dev/null 2>&1; then
            print_success "✅ Database is ready"
            break
        fi
        echo -n "."
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Database failed to start within timeout"
        exit 1
    fi
    
    # Wait for backend
    print_status "Waiting for backend API..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8080/health >/dev/null 2>&1; then
            print_success "✅ Backend API is ready"
            break
        fi
        echo -n "."
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Backend API might still be starting up (downloading models)"
    fi
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            print_success "✅ Frontend is ready"
            break
        fi
        echo -n "."
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Frontend might still be starting up"
    fi
}

# Function to show final status
show_final_status() {
    print_header "🎉 FaceSocial Platform Setup Complete!"
    echo ""
    echo "🌐 Access URLs:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🖥️  Frontend Application:  http://localhost:3000"
    echo "🔧 Backend API:           http://localhost:8080"
    echo "📚 API Documentation:     http://localhost:8080/docs"
    echo "🗄️  Database:             localhost:5432"
    echo "🗄️  Redis Cache:          localhost:6379"
    echo ""
    echo "👤 Default Login Credentials:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "👨‍💼 Admin:     username: admin,     password: admin123"
    echo "👤 Demo User:  username: demo_user, password: demo123"
    echo ""
    echo "🛠️  Management Commands:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Monitor System:        ./monitor-system.sh"
    echo "📋 View Logs:             docker-compose logs -f [service]"
    echo "🔄 Restart Services:      docker-compose restart"
    echo "🛑 Stop All Services:     docker-compose down"
    echo "🧹 Clean Everything:      docker-compose down -v --rmi all"
    echo ""
    print_success "✨ Platform is ready to use! Visit http://localhost:3000 to get started."
}

# Main execution
main() {
    clear
    print_header "╔══════════════════════════════════════════════════════════════════╗"
    print_header "║                   🚀 FaceSocial Platform Setup                   ║"
    print_header "║                      One-Click Installation                      ║"
    print_header "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    
    print_status "Starting automated setup process..."
    echo ""
    
    # Run all setup steps
    check_docker
    check_nvidia_docker
    setup_environment
    create_docker_compose
    setup_database_init
    create_monitoring_script
    start_services
    wait_for_services
    show_final_status
    
    echo ""
    print_header "🎯 Next Steps:"
    echo "1. Visit http://localhost:3000 to access the platform"
    echo "2. Login with admin/admin123 or demo_user/demo123"
    echo "3. Try the face recognition features"
    echo "4. Check the API documentation at http://localhost:8080/docs"
    echo ""
    print_status "Setup completed successfully! 🎉"
}

# Run main function
main "$@"
