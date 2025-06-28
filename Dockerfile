# Face Recognition System - Docker Image  
# Base: NVIDIA CUDA 12.9.1 with cuDNN on Ubuntu 22.04 (latest version with cuDNN 9+)
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDNN_VERSION=9
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    pkg-config \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    python3-pip \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake from official source
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# cuDNN is already included in the base image nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
# No additional cuDNN installation needed

# Create symbolic links for Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip and upgrade
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Copy requirements file first for better caching
COPY requirements-docker.txt ./requirements.txt

# Install Python dependencies in stages for better caching and debugging

# Stage 1: Install basic dependencies and cmake first
RUN python3.11 -m pip install --no-cache-dir \
    cmake==4.0.3 \
    wheel==0.45.1 \
    setuptools==78.1.1 \
    numpy==1.26.4

# Stage 2: Install PyTorch with CUDA support (compatible with CUDA 12.9)
# Note: Using cu121 as cu129 is not yet available, CUDA 12.9 is backward compatible
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Stage 3: Install ONNX Runtime GPU with cuDNN 9+ support (latest version)
RUN python3.11 -m pip install --no-cache-dir \
    onnxruntime-gpu==1.20.1

# Stage 4: Skip dlib for now - install other packages first  
# Note: dlib will be handled separately or via conda if needed

# Stage 5: Install TensorFlow (regular version only)
RUN python3.11 -m pip install --no-cache-dir tensorflow==2.19.0

# Stage 6: Handle blinker conflict - comprehensive removal and reinstall
# Remove ALL traces of blinker from system and site-packages
RUN apt-get update && \
    apt-get remove -y python3-blinker python3.11-blinker 2>/dev/null || true && \
    apt-get autoremove -y && \
    find /usr -name "*blinker*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /usr -name "*blinker*" -type f -delete 2>/dev/null || true && \
    find /home -name "*blinker*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /home -name "*blinker*" -type f -delete 2>/dev/null || true && \
    rm -rf /usr/lib/python3*/dist-packages/*blinker* 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3*/dist-packages/*blinker* 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3*/site-packages/*blinker* 2>/dev/null || true && \
    rm -rf /root/.local/lib/python3*/site-packages/*blinker* 2>/dev/null || true && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install blinker with correct version - use specific version to ensure compatibility
RUN python3.11 -m pip install --no-cache-dir --ignore-installed --force-reinstall "blinker==1.9.0"

# Stage 7: Install Flask and other web dependencies (AFTER blinker fix)
RUN python3.11 -m pip install --no-cache-dir \
    flask==3.1.1 \
    flask-cors==6.0.1 \
    werkzeug==3.1.3

# Stage 8: Install remaining packages from requirements (skip already installed ones)
# Create a temporary requirements file excluding already installed packages
RUN grep -v -E "^(flask|flask-cors|werkzeug|blinker|torch|torchvision|torchaudio|tensorflow|numpy|cmake|wheel|setuptools|onnxruntime-gpu).*" requirements.txt > temp_requirements.txt || cp requirements.txt temp_requirements.txt && \
    python3.11 -m pip install --no-cache-dir -r temp_requirements.txt && \
    rm temp_requirements.txt

# Stage 9: Create necessary directories
RUN mkdir -p /app/model/face-detection /app/model/face-recognition /app/output/detection /app/output/recognition /app/output/analysis /app/logs /app/temp /app/scripts

# Copy application code
COPY src/ ./src/
COPY *.py ./
COPY *.json ./
COPY *.md ./

# Copy scripts for model downloading
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x ./scripts/docker_model_setup.sh ./scripts/docker_startup.sh

# Try to copy model files if they exist locally
RUN if [ -d "model" ]; then cp -r model/* ./model/ 2>/dev/null || true; fi

# Clean up Python cache files and set permissions
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -name "*.pyo" -delete 2>/dev/null || true && \
    chmod +x /app/src/main.py

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command with model download support
CMD ["./scripts/docker_startup.sh", "python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
