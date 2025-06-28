#!/bin/bash
# Docker Startup Script with Model Download
# This script runs before starting the main application

echo "🚀 Starting Face Recognition System..."

# Check if running as root and switch to appuser if needed
if [ "$(id -u)" = "0" ]; then
    echo "⚠️  Running as root, switching to appuser..."
    exec su-exec appuser "$0" "$@"
fi

echo "👤 Running as user: $(whoami)"
echo "📂 Working directory: $(pwd)"

# Set environment variables for model paths
export MODEL_DIR="/app/model"
export FACE_DETECTION_MODEL_DIR="$MODEL_DIR/face-detection"
export FACE_RECOGNITION_MODEL_DIR="$MODEL_DIR/face-recognition"

echo "🔍 Checking model files..."

# Check if any model files are missing
missing_models=()

# Face Detection Models
if [ ! -f "$FACE_DETECTION_MODEL_DIR/yolov11n-face.onnx" ]; then
    missing_models+=("yolov11n-face.onnx")
fi
if [ ! -f "$FACE_DETECTION_MODEL_DIR/yolov11m-face.pt" ]; then
    missing_models+=("yolov11m-face.pt")
fi
if [ ! -f "$FACE_DETECTION_MODEL_DIR/yolov9c-face-lindevs.onnx" ]; then
    missing_models+=("yolov9c-face-lindevs.onnx")
fi

# Face Recognition Models
if [ ! -f "$FACE_RECOGNITION_MODEL_DIR/facenet_vggface2.onnx" ]; then
    missing_models+=("facenet_vggface2.onnx")
fi
if [ ! -f "$FACE_RECOGNITION_MODEL_DIR/arcface_r100.onnx" ]; then
    missing_models+=("arcface_r100.onnx")
fi
if [ ! -f "$FACE_RECOGNITION_MODEL_DIR/adaface_ir101.onnx" ]; then
    missing_models+=("adaface_ir101.onnx")
fi

# Count missing models
missing_count=${#missing_models[@]}

echo "📊 Model Status:"
echo "  - Total missing models: $missing_count"

if [ $missing_count -gt 0 ]; then
    echo "📥 Missing models detected, starting download process..."
    echo "  Missing: ${missing_models[*]}"
    
    # Run model download script
    if [ -f "/app/scripts/docker_model_setup.sh" ]; then
        echo "🔧 Running model download script..."
        bash /app/scripts/docker_model_setup.sh
        
        if [ $? -eq 0 ]; then
            echo "✅ Model download completed successfully"
        else
            echo "⚠️  Model download had some issues, but continuing startup"
            echo "ℹ️  The application may work with available models"
        fi
    else
        echo "❌ Model download script not found at /app/scripts/docker_model_setup.sh"
        echo "⚠️  Continuing startup without downloading models"
    fi
else
    echo "✅ All required models are present"
fi

# List available models
echo ""
echo "📋 Available Models:"
find "$MODEL_DIR" -name "*.onnx" -o -name "*.pt" | sort | while read -r model_file; do
    model_name=$(basename "$model_file")
    model_size=$(du -h "$model_file" 2>/dev/null | cut -f1)
    echo "  ✅ $model_name ($model_size)"
done

echo ""
echo "🎯 Starting application with command: $@"

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected, running in CPU mode"
fi

echo ""
echo "🚀 Launching Face Recognition System..."

# Execute the main command
exec "$@"
