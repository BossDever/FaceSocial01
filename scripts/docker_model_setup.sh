#!/bin/bash
# Docker Model Setup Script
# This script is used in Docker containers to download AI models if they don't exist

set -e  # Exit on any error

echo "🐳 Docker Model Setup Script Starting..."

# Model directory inside container
MODEL_DIR="/app/model"
SCRIPT_DIR="/app/scripts"

# Ensure model directories exist
mkdir -p "$MODEL_DIR/face-detection"
mkdir -p "$MODEL_DIR/face-recognition"

echo "📂 Model directories created:"
echo "  - $MODEL_DIR/face-detection"
echo "  - $MODEL_DIR/face-recognition"

# Function to check if file exists and has reasonable size
check_file() {
    local file_path="$1"
    local min_size_mb="$2"
    
    if [ ! -f "$file_path" ]; then
        echo "❌ File missing: $(basename "$file_path")"
        return 1
    fi
    
    local file_size=$(stat -c%s "$file_path" 2>/dev/null || echo "0")
    local file_size_mb=$((file_size / 1024 / 1024))
    
    if [ "$file_size_mb" -lt "$min_size_mb" ]; then
        echo "⚠️  File too small: $(basename "$file_path") (${file_size_mb}MB < ${min_size_mb}MB)"
        return 1
    fi
    
    echo "✅ File OK: $(basename "$file_path") (${file_size_mb}MB)"
    return 0
}

# Function to download file using wget (more reliable in Docker)
download_file() {
    local filename="$1"
    local file_id="$2"
    local folder="$3"
    local min_size="$4"
    
    local file_path="$MODEL_DIR/$folder/$filename"
    local download_url="https://drive.google.com/uc?export=download&id=$file_id"
    
    if check_file "$file_path" "$min_size"; then
        return 0
    fi
    
    echo "📥 Downloading $filename..."
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    local temp_file="$temp_dir/$filename"
    
    # Download with wget (better for Docker environments)
    if wget --no-check-certificate --quiet --load-cookies /tmp/cookies.txt \
           "https://drive.google.com/uc?export=download&confirm=\$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=$file_id' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" \
           -O "$temp_file"; then
        
        # Check if download was successful
        if check_file "$temp_file" "$min_size"; then
            mv "$temp_file" "$file_path"
            echo "✅ Successfully downloaded $filename"
        else
            echo "❌ Download failed or incomplete: $filename"
            rm -f "$temp_file"
            return 1
        fi
    else
        echo "❌ Download failed: $filename"
        rm -f "$temp_file"
        return 1
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    rm -f /tmp/cookies.txt
    
    return 0
}

# Model configurations (filename, file_id, folder, min_size_mb)
declare -a models=(
    "yolov11n-face.onnx:1JHXI0KILRSVTZnErTovuFGBoP-BKfZqS:face-detection:5"
    "yolov11m-face.pt:1GQFv9zLGdBkD4JCLeErkBfNylvCOlqAP:face-detection:35"
    "yolov9e-face-lindevs.onnx:1z1PbZHoHlcPnuJVKWJbpJ9OBlJ72yo2t:face-detection:200"
    "yolov9c-face-lindevs.onnx:1KJfaOv4Kbx0pw-VZWtgV_F1agPaVert9:face-detection:45"
    "facenet_vggface2.onnx:1E2uMDKxXwQVdZIEWgaLp5Fr9ag6zUvMv:face-recognition:80"
    "arcface_r100.onnx:1PNbmgXy4bNHOd0xBW35zMG7fjSeIOhBp:face-recognition:200"
    "adaface_ir101.onnx:1JWIroxhpcIRZ5OktTPgUT7LEAOrE9b_7:face-recognition:150"
)

echo "🚀 Starting model download process..."

success_count=0
total_count=${#models[@]}

# Download each model
for model_config in "${models[@]}"; do
    IFS=':' read -r filename file_id folder min_size <<< "$model_config"
    echo "📋 Processing $filename..."
    
    if download_file "$filename" "$file_id" "$folder" "$min_size"; then
        ((success_count++))
    fi
done

echo ""
echo "📊 Download Summary: $success_count/$total_count files successful"

if [ "$success_count" -eq "$total_count" ]; then
    echo "🎉 All models downloaded successfully!"
    exit 0
else
    echo "⚠️  Some models failed to download, but continuing..."
    echo "ℹ️  Available models:"
    find "$MODEL_DIR" -name "*.onnx" -o -name "*.pt" | while read -r file; do
        echo "  ✅ $(basename "$file")"
    done
    exit 0  # Don't fail container startup
fi
