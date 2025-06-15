#!/bin/bash
# Multi-Framework Face Recognition Installation Script
# สคริปต์ติดตั้งระบบจดจำใบหน้าแบบหลายเฟรมเวิร์ก

set -e  # Exit on any error

echo "🚀 Starting Multi-Framework Face Recognition Installation..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.8"

if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python version $python_version is compatible"
else
    print_error "Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Running in virtual environment: $VIRTUAL_ENV"
else
    print_warning "Not in a virtual environment. Consider creating one:"
    print_info "  python -m venv face_recognition_env"
    print_info "  source face_recognition_env/bin/activate  # Linux/Mac"
    print_info "  face_recognition_env\\Scripts\\activate  # Windows"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip

# Install system dependencies (if on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_info "Installing system dependencies (Linux)..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y cmake libopenblas-dev liblapack-dev
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
        sudo apt-get install -y build-essential
        print_status "Linux system dependencies installed"
    else
        print_warning "apt-get not found. Please install dependencies manually:"
        print_info "  cmake, libopenblas-dev, liblapack-dev, libgl1-mesa-glx, libglib2.0-0"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    print_info "macOS detected. Installing dependencies with Homebrew..."
    if command -v brew &> /dev/null; then
        brew install cmake openblas lapack
        print_status "macOS dependencies installed"
    else
        print_warning "Homebrew not found. Please install cmake, openblas, and lapack manually"
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    print_info "Windows detected. Please ensure Visual Studio Build Tools and CMake are installed"
    print_info "Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
fi

# Install core dependencies first
print_info "Installing core dependencies..."
pip install wheel cmake
pip install numpy opencv-python pillow
print_status "Core dependencies installed"

# Install PyTorch (with CUDA support if available)
print_info "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_info "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi
print_status "PyTorch installed"

# Install TensorFlow
print_info "Installing TensorFlow..."
pip install tensorflow==2.16.1
print_status "TensorFlow installed"

# Install dlib (this might take a while)
print_info "Installing dlib (this may take several minutes)..."
pip install dlib
print_status "Dlib installed"

# Install remaining requirements
print_info "Installing remaining requirements..."
pip install -r requirements.txt
print_status "All requirements installed"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p model/face-recognition
mkdir -p model/face-detection
mkdir -p output/analysis
mkdir -p output/detection
mkdir -p output/recognition
mkdir -p logs
mkdir -p temp
print_status "Directories created"

# Download essential models
print_info "Downloading essential models..."

# DeepFace models will be downloaded automatically on first use
print_info "DeepFace models will be downloaded automatically on first use"

# Download dlib models if not present
if [ ! -f "model/face-recognition/shape_predictor_5_face_landmarks.dat" ]; then
    print_info "Downloading dlib shape predictor..."
    wget -O model/face-recognition/shape_predictor_5_face_landmarks.dat.bz2 \
        "https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2"
    bzip2 -d model/face-recognition/shape_predictor_5_face_landmarks.dat.bz2
    print_status "Dlib shape predictor downloaded"
fi

if [ ! -f "model/face-recognition/dlib_face_recognition_resnet_model_v1.dat" ]; then
    print_info "Downloading dlib face recognition model..."
    wget -O model/face-recognition/dlib_face_recognition_resnet_model_v1.dat.bz2 \
        "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
    bzip2 -d model/face-recognition/dlib_face_recognition_resnet_model_v1.dat.bz2
    print_status "Dlib face recognition model downloaded"
fi

# Test installation
print_info "Testing installation..."
python -c "
import sys
import traceback

def test_import(module_name, package_name=None):
    try:
        if package_name:
            __import__(package_name)
        else:
            __import__(module_name)
        print(f'✅ {module_name} imported successfully')
        return True
    except ImportError as e:
        print(f'❌ {module_name} import failed: {e}')
        return False
    except Exception as e:
        print(f'⚠️ {module_name} import warning: {e}')
        return True

print('Testing framework imports...')
success_count = 0
total_count = 0

frameworks = [
    ('OpenCV', 'cv2'),
    ('NumPy', 'numpy'),
    ('PyTorch', 'torch'),
    ('TensorFlow', 'tensorflow'),
    ('DeepFace', 'deepface'),
    ('FaceNet-PyTorch', 'facenet_pytorch'),
    ('Dlib', 'dlib'),
    ('InsightFace', 'insightface'),
    ('Pillow', 'PIL'),
    ('FastAPI', 'fastapi'),
    ('Uvicorn', 'uvicorn'),
]

for name, module in frameworks:
    total_count += 1
    if test_import(name, module):
        success_count += 1

print(f'\n📊 Import Test Results: {success_count}/{total_count} frameworks available')

if success_count >= 7:  # Core frameworks
    print('🎉 Installation successful! Core frameworks are available.')
else:
    print('⚠️ Some frameworks failed to import. Check the error messages above.')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Installation test passed!"
else
    print_error "Installation test failed!"
    exit 1
fi

# Final instructions
echo ""
echo "🎉 Multi-Framework Face Recognition Installation Complete!"
echo "========================================================"
echo ""
print_info "Available frameworks:"
print_info "  ✓ DeepFace (VGG-Face, FaceNet, ArcFace, Dlib, SFace)"
print_info "  ✓ FaceNet-PyTorch (VGGFace2, CASIA-WebFace)"
print_info "  ✓ Dlib (ResNet-based face recognition)"
print_info "  ✓ InsightFace (ArcFace, various models)"
print_info "  ✓ Custom EdgeFace implementation"
echo ""
print_info "Next steps:"
print_info "  1. Start the server: python start.py"
print_info "  2. Test the installation: python test_multi_framework.py --test-type single"
print_info "  3. Run comprehensive tests: python test_multi_framework.py --test-type all"
print_info "  4. Access the API documentation at: http://localhost:8000/docs"
echo ""
print_info "For GPU acceleration, ensure CUDA is properly installed:"
print_info "  - NVIDIA CUDA Toolkit 11.8 or compatible"
print_info "  - cuDNN library"
print_info "  - Compatible GPU drivers"
echo ""
print_status "Happy face recognizing! 🎭"
