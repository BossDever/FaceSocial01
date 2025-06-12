#!/usr/bin/env python3
"""
Setup script for Face Recognition System
"""

import os
import sys
import subprocess
from typing import List

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_cuda() -> bool:
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU mode")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
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
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

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
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model_name}: {model_path} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {model_name}: {model_path} (NOT FOUND)")
            missing_models.append(model_name)
    
    return missing_models

def install_requirements() -> bool:
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def test_imports() -> List[str]:
    """Test if critical imports work"""
    critical_imports = [
        "fastapi",
        "uvicorn", 
        "torch",
        "cv2",
        "numpy",
        "onnxruntime"
    ]
    
    failed_imports = []
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return failed_imports

def main() -> bool:
    """Main setup function"""
    print("🚀 Face Recognition System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        return False
    
    # Test imports
    print("\n🧪 Testing imports...")
    failed_imports = test_imports()
    
    # Check CUDA
    print("\n🔥 Checking CUDA...")
    check_cuda()
    
    # Check model files
    print("\n🤖 Checking model files...")
    missing_models = check_model_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
        print("   Try: pip install -r requirements.txt")
        return False
    
    if missing_models:
        print(f"⚠️ Missing models: {', '.join(missing_models)}")
        print("   Place model files in the model/ directory")
        print("   System will still work but with limited functionality")
    else:
        print("✅ All model files found!")
    
    print("\n🎉 Setup completed!")
    print("\n🚀 To start the system:")
    print("   python src/main.py")
    print("   or")
    print("   uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
