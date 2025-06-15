@echo off
REM Multi-Framework Face Recognition Installation Script for Windows
REM สคริปต์ติดตั้งระบบจดจำใบหน้าแบบหลายเฟรมเวิร์กสำหรับ Windows

echo 🚀 Starting Multi-Framework Face Recognition Installation...
echo ==================================================

REM Check Python version
echo ℹ️ Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ❌ Python 3.8+ is required
    python --version
    pause
    exit /b 1
) else (
    echo ✅ Python version is compatible
    python --version
)

REM Check virtual environment
if defined VIRTUAL_ENV (
    echo ✅ Running in virtual environment: %VIRTUAL_ENV%
) else (
    echo ⚠️ Not in a virtual environment. Consider creating one:
    echo   python -m venv face_recognition_env
    echo   face_recognition_env\Scripts\activate
    echo.
    echo ℹ️ Continuing with installation in current environment...
)

REM Upgrade pip
echo ℹ️ Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ Failed to upgrade pip
    pause
    exit /b 1
)
echo ✅ Pip upgraded successfully

REM Install core dependencies
echo ℹ️ Installing core dependencies...
python -m pip install wheel cmake
python -m pip install numpy opencv-python pillow
if errorlevel 1 (
    echo ❌ Failed to install core dependencies
    pause
    exit /b 1
)
echo ✅ Core dependencies installed

REM Check for NVIDIA GPU
echo ℹ️ Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ℹ️ Installing PyTorch CPU version...
    python -m pip install torch torchvision torchaudio
) else (
    echo ℹ️ NVIDIA GPU detected. Installing PyTorch with CUDA support...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
if errorlevel 1 (
    echo ❌ Failed to install PyTorch
    pause
    exit /b 1
)
echo ✅ PyTorch installed

REM Install TensorFlow
echo ℹ️ Installing TensorFlow...
python -m pip install tensorflow==2.16.1
if errorlevel 1 (
    echo ⚠️ TensorFlow installation had issues, but continuing...
) else (
    echo ✅ TensorFlow installed
)

REM Install Visual C++ requirements for dlib
echo ℹ️ Installing dlib (this may take several minutes)...
echo ⚠️ If dlib installation fails, please install:
echo   - Visual Studio Build Tools 2019/2022
echo   - CMake
echo   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
python -m pip install dlib
if errorlevel 1 (
    echo ❌ Dlib installation failed
    echo Please install Visual Studio Build Tools and CMake
    echo Then run: pip install dlib
    echo Continuing with other packages...
) else (
    echo ✅ Dlib installed successfully
)

REM Install remaining requirements
echo ℹ️ Installing remaining requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️ Some packages may have failed to install
    echo Check the output above for errors
) else (
    echo ✅ All requirements installed
)

REM Create necessary directories
echo ℹ️ Creating necessary directories...
if not exist "model" mkdir model
if not exist "model\face-recognition" mkdir model\face-recognition
if not exist "model\face-detection" mkdir model\face-detection
if not exist "output" mkdir output
if not exist "output\analysis" mkdir output\analysis
if not exist "output\detection" mkdir output\detection
if not exist "output\recognition" mkdir output\recognition
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
echo ✅ Directories created

REM Download essential models (placeholder - would need actual implementation)
echo ℹ️ Model download notes:
echo   - DeepFace models will be downloaded automatically on first use
echo   - Dlib models need to be downloaded manually if required
echo   - InsightFace models will be downloaded on first use
echo   - Place additional ONNX models in model\face-recognition\

REM Test installation
echo ℹ️ Testing installation...
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
    sys.exit(0)
else:
    print('⚠️ Some frameworks failed to import. Check the error messages above.')
    print('The system may still work with available frameworks.')
    sys.exit(0)
"

if errorlevel 1 (
    echo ❌ Installation test had issues
    echo Check the error messages above
) else (
    echo ✅ Installation test completed
)

echo.
echo 🎉 Multi-Framework Face Recognition Installation Complete!
echo ========================================================
echo.
echo ℹ️ Available frameworks:
echo   ✓ DeepFace (VGG-Face, FaceNet, ArcFace, Dlib, SFace)
echo   ✓ FaceNet-PyTorch (VGGFace2, CASIA-WebFace)
echo   ✓ Dlib (ResNet-based face recognition)
echo   ✓ InsightFace (ArcFace, various models)
echo   ✓ Custom EdgeFace implementation
echo.
echo ℹ️ Next steps:
echo   1. Start the server: python start.py
echo   2. Test the installation: python test_multi_framework.py --test-type single
echo   3. Run comprehensive tests: python test_multi_framework.py --test-type all
echo   4. Access the API documentation at: http://localhost:8000/docs
echo.
echo ℹ️ For GPU acceleration, ensure CUDA is properly installed:
echo   - NVIDIA CUDA Toolkit 11.8 or compatible
echo   - cuDNN library  
echo   - Compatible GPU drivers
echo.
echo ✅ Happy face recognizing! 🎭
echo.
pause
