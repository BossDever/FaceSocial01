# ชุดคำสั่ง: Docker & Deployment
## การ Deploy ระบบด้วย Docker และ Container

### 📋 สารบัญ
9.1 [ภาพรวม Docker Deployment](#91-ภาพรวม-docker-deployment)
9.2 [Docker Configuration](#92-docker-configuration)
9.3 [Docker Compose Setup](#93-docker-compose-setup)
9.4 [Environment Management](#94-environment-management)
9.5 [GPU Configuration](#95-gpu-configuration)
9.6 [Database Setup](#96-database-setup)
9.7 [Frontend Deployment](#97-frontend-deployment)
9.8 [Production Deployment](#98-production-deployment)

---

## 9.1 ภาพรวม Docker Deployment

ระบบ Face Recognition Platform สามารถ deploy ได้หลายรูปแบบ:
- **Development**: Docker Compose สำหรับ local development
- **Production**: Kubernetes หรือ Docker Swarm
- **Cloud**: AWS ECS, Google Cloud Run, Azure Container Instances

### 🏗️ สถาปัตยกรรม Deployment
```
Production Environment
├── Backend (Python/FastAPI + GPU)
├── Frontend (Next.js/React)
├── Database (PostgreSQL)
├── Redis (Caching)
├── Nginx (Load Balancer)
└── File Storage (S3/MinIO)
```

### 🎯 ข้อกำหนดระบบ
- **GPU**: NVIDIA GPU with CUDA 12.9+ support
- **Memory**: 8GB+ RAM, 4GB+ VRAM
- **Storage**: 10GB+ for models and data
- **Network**: High-speed internet for model downloads

---

## 9.2 Docker Configuration

### 9.2.1 Backend Dockerfile
```dockerfile
# Face Recognition System - Docker Image  
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDNN_VERSION=9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget curl git build-essential pkg-config \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv \
    python3-pip \
    libopencv-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libfontconfig1 \
    libgomp1 ffmpeg \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module \
    libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python and create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
COPY requirements-docker.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs model/face-detection model/face-recognition \
    uploads temp

# Set permissions
RUN chmod +x start.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["python", "start.py"]
```

### 9.2.2 Frontend Dockerfile
```dockerfile
# Frontend Dockerfile (Next.js)
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json package-lock.json* ./
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build application
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

### 9.2.3 Build Scripts
```bash
#!/bin/bash
# docker-build.sh - Backend build script

echo "🐳 Building Face Recognition Backend..."

# Build backend image
docker build \
    --tag face-recognition-backend:latest \
    --tag face-recognition-backend:$(date +%Y%m%d-%H%M%S) \
    --build-arg CUDA_VERSION=12.9.1 \
    --build-arg PYTHON_VERSION=3.11 \
    .

echo "✅ Backend build completed"

# Build frontend image
echo "🐳 Building Face Recognition Frontend..."

cd projec-final-fronend

docker build \
    --tag face-recognition-frontend:latest \
    --tag face-recognition-frontend:$(date +%Y%m%d-%H%M%S) \
    .

echo "✅ Frontend build completed"

cd ..

echo "🎉 All images built successfully"
```

---

## 9.3 Docker Compose

### 9.3.1 Development Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Backend API
  face-recognition-api:
    build: 
      context: .
      dockerfile: Dockerfile
    image: face-recognition-backend:latest
    container_name: face-recognition-container
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/face_recognition
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8080:8080"
    volumes:
      - ./model:/app/model
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./temp:/app/temp
    depends_on:
      - postgres
      - redis
    networks:
      - face-recognition-network

  # Frontend
  face-recognition-frontend:
    build:
      context: ./projec-final-fronend
      dockerfile: Dockerfile
    image: face-recognition-frontend:latest
    container_name: face-recognition-frontend
    restart: unless-stopped
    environment:
      - NODE_ENV=development
      - NEXT_PUBLIC_API_URL=http://localhost:8080
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/face_recognition_frontend
    ports:
      - "3000:3000"
    volumes:
      - ./projec-final-fronend:/app
      - /app/node_modules
    depends_on:
      - face-recognition-api
      - postgres
    networks:
      - face-recognition-network

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: face-recognition-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=face_recognition
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_MULTIPLE_DATABASES=face_recognition_frontend
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./projec-final-fronend/database/init:/docker-entrypoint-initdb.d
    networks:
      - face-recognition-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: face-recognition-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - face-recognition-network

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    container_name: face-recognition-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./projec-final-fronend/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./projec-final-fronend/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - face-recognition-api
      - face-recognition-frontend
    networks:
      - face-recognition-network

volumes:
  postgres_data:
  redis_data:

networks:
  face-recognition-network:
    driver: bridge
```

### 9.3.2 Production Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  face-recognition-api:
    image: face-recognition-backend:latest
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - model_data:/app/model
      - uploads_data:/app/uploads
      - logs_data:/app/logs
    networks:
      - face-recognition-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  face-recognition-frontend:
    image: face-recognition-frontend:latest
    restart: always
    deploy:
      replicas: 2
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=${API_URL}
      - DATABASE_URL=${FRONTEND_DATABASE_URL}
    networks:
      - face-recognition-network

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/prod.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - static_files:/var/www/static
    depends_on:
      - face-recognition-api
      - face-recognition-frontend
    networks:
      - face-recognition-network

volumes:
  model_data:
  uploads_data:
  logs_data:
  static_files:

networks:
  face-recognition-network:
    driver: overlay
```

---

## 9.4 Environment Setup

### 9.4.1 Environment Variables
```bash
# .env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/face_recognition
FRONTEND_DATABASE_URL=postgresql://postgres:password@localhost:5432/face_recognition_frontend

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=1

# AI Models
MODEL_PATH=/app/model
ENABLE_GPU=true
MAX_VRAM_USAGE=0.8

# File Upload
UPLOAD_PATH=/app/uploads
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_UPLOAD_MAX_SIZE=10485760
```

```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# Database (Production)
DATABASE_URL=${PRODUCTION_DATABASE_URL}
FRONTEND_DATABASE_URL=${FRONTEND_PRODUCTION_DATABASE_URL}
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis (Production)
REDIS_URL=${PRODUCTION_REDIS_URL}
REDIS_POOL_SIZE=10

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4

# AI Models
MODEL_PATH=/app/model
ENABLE_GPU=true
MAX_VRAM_USAGE=0.7
MODEL_CACHE_SIZE=3

# File Storage (S3)
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_S3_BUCKET=${AWS_S3_BUCKET}
AWS_S3_REGION=${AWS_S3_REGION}

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30
```

### 9.4.2 Environment Management Script
```bash
#!/bin/bash
# setup-environment.sh

set -e

echo "🔧 Setting up Face Recognition Platform environment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check NVIDIA Docker (for GPU support)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    
    if ! docker info | grep -q nvidia; then
        echo "⚠️  NVIDIA Docker runtime not detected. Installing..."
        # Add NVIDIA Docker installation commands here
    fi
else
    echo "⚠️  No NVIDIA GPU detected. Running in CPU mode."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p {model,uploads,logs,temp}
mkdir -p projec-final-fronend/{uploads,public/temp}

# Download models if not exists
echo "📦 Checking AI models..."
if [ ! -f "model/face-detection/yolov11m-face.pt" ]; then
    echo "📥 Downloading face detection model..."
    # Add model download commands here
fi

if [ ! -f "model/face-recognition/adaface_ir101.onnx" ]; then
    echo "📥 Downloading face recognition model..."
    # Add model download commands here
fi

# Set up database
echo "🗄️  Setting up database..."
./setup-database.sh

echo "✅ Environment setup completed!"
echo ""
echo "🚀 To start the application:"
echo "   Development: docker-compose up -d"
echo "   Production:  docker-compose -f docker-compose.prod.yml up -d"
```

---

## 9.5 GPU Configuration

### 9.5.1 NVIDIA Docker Setup
```bash
#!/bin/bash
# setup-nvidia-docker.sh

echo "🎮 Setting up NVIDIA Docker support..."

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

# Install NVIDIA Container Toolkit
echo "📦 Installing NVIDIA Container Toolkit..."

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
echo "🔄 Restarting Docker daemon..."
sudo systemctl restart docker

# Test NVIDIA Docker
echo "🧪 Testing NVIDIA Docker..."
if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi; then
    echo "✅ NVIDIA Docker setup successful!"
else
    echo "❌ NVIDIA Docker test failed"
    exit 1
fi

echo "✅ GPU configuration completed!"
```

### 9.5.2 GPU Configuration in Docker
```yaml
# GPU-specific configuration
services:
  face-recognition-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Use 1 GPU
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
```

### 9.5.3 GPU Monitoring
```bash
#!/bin/bash
# monitor-gpu.sh

echo "📊 GPU Monitoring Dashboard"
echo "=========================="

while true; do
    clear
    echo "📊 GPU Status - $(date)"
    echo "=========================="
    
    # NVIDIA GPU status
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
        echo ""
        
        # Check Docker containers using GPU
        echo "🐳 Containers using GPU:"
        docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" --no-stream | head -5
    else
        echo "❌ No NVIDIA GPU detected"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
```

---

## 9.6 Database Setup

### 9.6.1 Database Initialization
```bash
#!/bin/bash
# setup-database.sh

set -e

echo "🗄️ Setting up PostgreSQL database..."

# Database configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-password}
DB_NAME=${DB_NAME:-face_recognition}
DB_FRONTEND_NAME=${DB_FRONTEND_NAME:-face_recognition_frontend}

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
while ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Create databases if they don't exist
echo "📝 Creating databases..."

PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME"

PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_FRONTEND_NAME'" | grep -q 1 || \
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_FRONTEND_NAME"

# Run initialization scripts
echo "🔧 Running database initialization scripts..."

# Backend database schema
if [ -f "projec-final-fronend/database/init/01_schema.sql" ]; then
    echo "📄 Running backend schema..."
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f projec-final-fronend/database/init/01_schema.sql
fi

# Social features schema
if [ -f "projec-final-fronend/database/init/02_social_features.sql" ]; then
    echo "📄 Running social features schema..."
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f projec-final-fronend/database/init/02_social_features.sql
fi

# Frontend database schema (Prisma)
echo "🔧 Setting up frontend database with Prisma..."
cd projec-final-fronend

if [ -f "prisma/schema.prisma" ]; then
    echo "📄 Running Prisma migrations..."
    npx prisma migrate deploy
    npx prisma generate
fi

cd ..

echo "✅ Database setup completed!"
```

### 🔄 Database Backup Script
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

echo "💾 Creating database backup..."

# Backup main database
echo "📦 Backing up main database..."
PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME > "$BACKUP_DIR/face_recognition_$TIMESTAMP.sql"

# Backup frontend database
echo "📦 Backing up frontend database..."
PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_FRONTEND_NAME > "$BACKUP_DIR/face_recognition_frontend_$TIMESTAMP.sql"

# Compress backups
echo "🗜️  Compressing backups..."
gzip "$BACKUP_DIR/face_recognition_$TIMESTAMP.sql"
gzip "$BACKUP_DIR/face_recognition_frontend_$TIMESTAMP.sql"

# Clean old backups (keep last 7 days)
echo "🧹 Cleaning old backups..."
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "✅ Database backup completed!"
echo "📁 Backup files:"
echo "   - $BACKUP_DIR/face_recognition_$TIMESTAMP.sql.gz"
echo "   - $BACKUP_DIR/face_recognition_frontend_$TIMESTAMP.sql.gz"
```

---

## 9.7 Frontend Deployment

### ⚛️ Next.js Production Build
```bash
#!/bin/bash
# build-frontend.sh

echo "🏗️ Building Next.js frontend for production..."

cd projec-final-fronend

# Install dependencies
echo "📦 Installing dependencies..."
npm ci

# Build application
echo "🔨 Building application..."
npm run build

# Test build
echo "🧪 Testing build..."
npm run start &
SERVER_PID=$!

# Wait for server to start
sleep 10

# Check if server is responding
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Build test successful!"
else
    echo "❌ Build test failed!"
    exit 1
fi

# Stop test server
kill $SERVER_PID

echo "✅ Frontend build completed!"

cd ..
```

### 🌐 Nginx Configuration
```nginx
# nginx/prod.conf
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server face-recognition-api:8080;
    }
    
    upstream frontend {
        server face-recognition-frontend:3000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;
    
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        
        # Client max body size for file uploads
        client_max_body_size 50M;
        
        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts for AI processing
            proxy_connect_timeout 30s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        # Upload routes with stricter rate limiting
        location /api/upload {
            limit_req zone=upload burst=5 nodelay;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Extended timeouts for file uploads
            proxy_connect_timeout 60s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
        }
        
        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
        
        # Health check
        location /health {
            access_log off;
            proxy_pass http://backend;
        }
    }
}
```

---

## 9.8 Production Deployment

### 🚀 Production Deployment Script
```bash
#!/bin/bash
# deploy-production.sh

set -e

echo "🚀 Deploying Face Recognition Platform to Production..."

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.prod.yml}
BACKUP_BEFORE_DEPLOY=${BACKUP_BEFORE_DEPLOY:-true}

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if all required environment variables are set
required_vars=("DATABASE_URL" "REDIS_URL" "SECRET_KEY" "API_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Backup database before deployment
if [ "$BACKUP_BEFORE_DEPLOY" = "true" ]; then
    echo "💾 Creating pre-deployment backup..."
    ./backup-database.sh
fi

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose -f $COMPOSE_FILE pull

# Build new images
echo "🔨 Building application images..."
docker-compose -f $COMPOSE_FILE build --no-cache

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose -f $COMPOSE_FILE down

# Start services
echo "🚀 Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
timeout=300
elapsed=0

while [ $elapsed -lt $timeout ]; do
    if docker-compose -f $COMPOSE_FILE ps | grep -q "healthy\|Up"; then
        echo "✅ Services are healthy!"
        break
    fi
    
    echo "Waiting... ($elapsed/$timeout seconds)"
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ $elapsed -ge $timeout ]; then
    echo "❌ Services failed to become healthy within $timeout seconds"
    echo "📋 Service status:"
    docker-compose -f $COMPOSE_FILE ps
    exit 1
fi

# Run post-deployment tests
echo "🧪 Running post-deployment tests..."
./test-deployment.sh

echo "✅ Production deployment completed successfully!"
echo ""
echo "📊 Service Status:"
docker-compose -f $COMPOSE_FILE ps
echo ""
echo "🌐 Application URLs:"
echo "   Frontend: https://your-domain.com"
echo "   API:      https://your-domain.com/api"
echo "   Health:   https://your-domain.com/health"
```

### 🧪 Deployment Testing
```bash
#!/bin/bash
# test-deployment.sh

echo "🧪 Testing deployment..."

# Configuration
API_URL=${API_URL:-http://localhost:8080}
FRONTEND_URL=${FRONTEND_URL:-http://localhost:3000}

# Test API health
echo "🏥 Testing API health..."
if curl -f "$API_URL/health" > /dev/null 2>&1; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    exit 1
fi

# Test frontend
echo "🌐 Testing frontend..."
if curl -f "$FRONTEND_URL" > /dev/null 2>&1; then
    echo "✅ Frontend check passed"
else
    echo "❌ Frontend check failed"
    exit 1
fi

# Test database connection
echo "🗄️ Testing database connection..."
if curl -f "$API_URL/api/health/database" > /dev/null 2>&1; then
    echo "✅ Database connection test passed"
else
    echo "❌ Database connection test failed"
    exit 1
fi

# Test AI models
echo "🤖 Testing AI models..."
if curl -f "$API_URL/api/models/status" > /dev/null 2>&1; then
    echo "✅ AI models test passed"
else
    echo "❌ AI models test failed"
    exit 1
fi

# Test GPU (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Testing GPU access..."
    if curl -f "$API_URL/api/system/gpu" > /dev/null 2>&1; then
        echo "✅ GPU access test passed"
    else
        echo "⚠️ GPU access test failed (continuing anyway)"
    fi
fi

echo "✅ All deployment tests passed!"
```

### 📊 Monitoring Setup
```bash
#!/bin/bash
# setup-monitoring.sh

echo "📊 Setting up monitoring..."

# Prometheus configuration
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'face-recognition-api'
    static_configs:
      - targets: ['face-recognition-api:9090']
  
  - job_name: 'face-recognition-frontend'
    static_configs:
      - targets: ['face-recognition-frontend:9091']
      
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
EOF

# Grafana dashboard
cat > grafana-dashboard.json << EOF
{
  "dashboard": {
    "title": "Face Recognition Platform",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"
          }
        ]
      },
      {
        "title": "GPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu"
          }
        ]
      }
    ]
  }
}
EOF

echo "✅ Monitoring setup completed!"
```

---

## สรุป

การ Deploy ระบบ Face Recognition Platform ด้วย Docker ให้ความยืดหยุ่นและความเสถียร:

### ✅ ข้อดีของ Docker Deployment
- **Consistency**: ทำงานเหมือนกันทุกสภาพแวดล้อม
- **Scalability**: ขยายระบบได้ง่าย
- **Isolation**: แยกส่วนระหว่าง services
- **Rollback**: กลับไปเวอร์ชันเดิมได้รวดเร็ว
- **Monitoring**: ติดตามระบบได้ครบถ้วน

### 🎯 Best Practices
- ใช้ Multi-stage builds เพื่อลดขนาด image
- ตั้งค่า Health checks สำหรับทุก service
- ใช้ Secrets management สำหรับข้อมูลสำคัญ
- สำรองข้อมูลก่อน deploy ทุกครั้ง
- ติดตาม logs และ metrics

### 🚀 การใช้งาน
- **Development**: `docker-compose up -d`
- **Production**: `./deploy-production.sh`
- **Monitoring**: Access Grafana dashboard
- **Backup**: `./backup-database.sh`

ระบบ Docker deployment นี้พร้อมสำหรับ production และสามารถขยายไปยัง Kubernetes หรือ cloud platforms ได้ในอนาคต
