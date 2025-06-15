#!/bin/bash
# Quick Multi-Framework Installation for Git Bash/Windows
# สคริปต์ติดตั้งเร็วสำหรับ Git Bash

echo "🚀 Quick Multi-Framework Face Recognition Installation"
echo "=================================================="

# Check Python
echo "ℹ️ Checking Python..."
python --version

# Upgrade pip
echo "ℹ️ Upgrading pip..."
python -m pip install --upgrade pip

# Install core packages
echo "ℹ️ Installing core packages..."
pip install wheel cmake numpy opencv-python pillow

# Install PyTorch
echo "ℹ️ Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected - installing CUDA version"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing CPU version"
    pip install torch torchvision torchaudio
fi

# Install TensorFlow
echo "ℹ️ Installing TensorFlow..."
pip install tensorflow==2.16.1

# Install other ML frameworks
echo "ℹ️ Installing face recognition frameworks..."
pip install deepface
pip install facenet-pytorch
pip install insightface

# Try to install dlib (may fail on Windows without proper build tools)
echo "ℹ️ Attempting to install dlib..."
pip install dlib || echo "⚠️ Dlib installation failed - please install Visual Studio Build Tools"

# Install remaining requirements
echo "ℹ️ Installing remaining requirements..."
pip install fastapi uvicorn python-multipart pydantic
pip install aiofiles requests coloredlogs tqdm
pip install pandas scipy matplotlib seaborn
pip install scikit-learn imageio

# Create directories
echo "ℹ️ Creating directories..."
mkdir -p model/face-recognition
mkdir -p model/face-detection
mkdir -p output/{analysis,detection,recognition}
mkdir -p logs temp

# Test installation
echo "ℹ️ Testing installation..."
python -c "
frameworks = [
    ('OpenCV', 'cv2'),
    ('NumPy', 'numpy'), 
    ('PyTorch', 'torch'),
    ('TensorFlow', 'tensorflow'),
    ('DeepFace', 'deepface'),
    ('FaceNet-PyTorch', 'facenet_pytorch'),
    ('InsightFace', 'insightface'),
    ('FastAPI', 'fastapi'),
]

success = 0
for name, module in frameworks:
    try:
        __import__(module)
        print(f'✅ {name} - OK')
        success += 1
    except ImportError:
        print(f'❌ {name} - Failed')

print(f'\\n📊 Installation Results: {success}/{len(frameworks)} frameworks available')
if success >= 6:
    print('🎉 Installation successful!')
else:
    print('⚠️ Some frameworks failed - check error messages above')
"

echo ""
echo "🎉 Installation completed!"
echo "Next steps:"
echo "  1. python start.py"
echo "  2. python test_multi_framework.py --test-type single"
echo ""
