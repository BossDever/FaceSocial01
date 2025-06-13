#!/usr/bin/env python3
"""
Setup script for Face Recognition System
Enhanced version with proper error handling and comprehensive checks
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any


def check_python_version() -> bool:
    """Check if Python version is compatible"""
    try:
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True
    except Exception as e:
        print(f"❌ Error checking Python version: {e}")
        return False


def check_cuda() -> bool:
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA available: {device_count} device(s)")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"   Device {i}: {device_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU mode")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet - CUDA check skipped")
        return False
    except Exception as e:
        print(f"⚠️ Error checking CUDA: {e}")
        return False


def create_directories() -> None:
    """Create necessary directories"""
    directories = [
        "logs",
        "output",
        "output/detection", 
        "output/recognition",
        "output/analysis",
        "temp",
        "model",
        "model/face-detection",
        "model/face-recognition"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"⚠️ Failed to create {directory}: {e}")


def check_model_files() -> List[str]:
    """Check if model files exist"""
    model_files = {
        "YOLOv9c": "model/face-detection/yolov9c-face-lindevs.onnx",
        "YOLOv9e": "model/face-detection/yolov9e-face-lindevs.onnx", 
        "YOLOv11m": "model/face-detection/yolov11m-face.pt",
        "FaceNet": "model/face-recognition/facenet_vggface2.onnx",
        "AdaFace": "model/face-recognition/adaface_ir101.onnx",
        "ArcFace": "model/face-recognition/arcface_r100.onnx"
    }
    
    missing_models = []
    available_models = []
    
    print("🤖 Checking model files...")
    for model_name, model_path in model_files.items():
        if Path(model_path).exists():
            try:
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                print(f"✅ {model_name}: {model_path} ({size_mb:.1f}MB)")
                available_models.append(model_name)
            except Exception as e:
                print(f"⚠️ {model_name}: Error reading file - {e}")
                missing_models.append(model_name)
        else:
            print(f"❌ {model_name}: {model_path} (NOT FOUND)")
            missing_models.append(model_name)
    
    if available_models:
        print(f"✅ Available models: {len(available_models)}/{len(model_files)}")
    else:
        print("⚠️ No model files found - system will have limited functionality")
    
    return missing_models


def install_requirements() -> bool:
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("⬆️ Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], timeout=300)
        
        # Install requirements
        print("📦 Installing requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], timeout=1800)  # 30 minutes timeout
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("💡 Try manually: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during installation: {e}")
        return False


def test_imports() -> List[str]:
    """Test if critical imports work"""
    critical_imports = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"), 
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("onnxruntime", "ONNX Runtime"),
        ("pydantic", "Pydantic"),
        ("starlette", "Starlette")
    ]
    
    failed_imports = []
    print("🧪 Testing imports...")
    
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"✅ {description} ({module})")
        except ImportError as e:
            print(f"❌ {description} ({module}): {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"⚠️ {description} ({module}): Unexpected error - {e}")
            failed_imports.append(module)
    
    return failed_imports


def test_basic_functionality() -> bool:
    """Test basic system functionality"""
    print("🔧 Testing basic functionality...")
    
    try:
        # Test core imports
        from src.core.config import get_settings
        print("✅ Core config import successful")
        
        # Test settings
        settings = get_settings()
        print(f"✅ Settings loaded - Host: {settings.host}, Port: {settings.port}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def check_port_availability(port: int = 8080) -> bool:
    """Check if the default port is available"""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result == 0:
                print(f"⚠️ Port {port} is already in use")
                return False
            else:
                print(f"✅ Port {port} is available")
                return True
    except Exception as e:
        print(f"⚠️ Could not check port availability: {e}")
        return True  # Assume it's available


def create_startup_scripts() -> None:
    """Create convenient startup scripts if they don't exist"""
    scripts = {
        "start_dev.py": '''#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from start import main
if __name__ == "__main__":
    sys.argv.extend(["--reload", "--log-level", "DEBUG"])
    main()
''',
        "start_prod.py": '''#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from start import main
if __name__ == "__main__":
    sys.argv.extend(["--no-reload", "--log-level", "INFO"])
    main()
'''
    }
    
    for script_name, content in scripts.items():
        script_path = Path(script_name)
        if not script_path.exists():
            try:
                script_path.write_text(content, encoding='utf-8')
                print(f"✅ Created {script_name}")
            except Exception as e:
                print(f"⚠️ Failed to create {script_name}: {e}")


def print_summary(missing_models: List[str], failed_imports: List[str]) -> None:
    """Print setup summary"""
    print("\n" + "=" * 60)
    print("📋 Setup Summary")
    print("=" * 60)
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
        print("   💡 Try: pip install -r requirements.txt")
        print("   💡 Or check specific package installation")
    else:
        print("✅ All critical imports successful!")
    
    if missing_models:
        print(f"⚠️ Missing models: {', '.join(missing_models)}")
        print("   💡 Place model files in the model/ directory:")
        print("   📁 model/face-detection/ (for YOLO models)")
        print("   📁 model/face-recognition/ (for recognition models)")
        print("   ℹ️ System will work with limited functionality")
    else:
        print("✅ All model files found!")
    
    print("\n🎉 Setup completed!")
    print("\n🚀 To start the system:")
    print("   🖥️ Command line: python start.py")
    print("   🛠️ Development:  python start.py --reload")
    print("   🏭 Production:   python start.py --no-reload")
    print("   🌐 Web UI:       http://localhost:8080")
    print("   📚 API Docs:     http://localhost:8080/docs")


def main() -> bool:
    """Main setup function"""
    print("🚀 Face Recognition System Setup")
    print("=" * 50)
    
    setup_start_time = time.time()
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install requirements
    if not install_requirements():
        print("⚠️ Continuing with existing packages...")
    
    # Step 4: Test imports
    failed_imports = test_imports()
    
    # Step 5: Check CUDA
    check_cuda()
    
    # Step 6: Check model files
    missing_models = check_model_files()
    
    # Step 7: Test basic functionality
    if not failed_imports:
        test_basic_functionality()
    
    # Step 8: Check port availability
    check_port_availability()
    
    # Step 9: Create startup scripts
    create_startup_scripts()
    
    # Step 10: Print summary
    setup_time = time.time() - setup_start_time
    print_summary(missing_models, failed_imports)
    print(f"\n⏱️ Setup completed in {setup_time:.2f} seconds")
    
    # Return success status
    success = len(failed_imports) == 0
    if not success:
        print("\n⚠️ Setup completed with warnings - see above for details")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error during setup: {e}")
        sys.exit(1)