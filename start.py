#!/usr/bin/env python3
"""
Start script for Face Recognition System
Comprehensive startup with health checks and configuration
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/startup.log")
        ]
    )
    return logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements"""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error(f"❌ Python 3.8+ required, found {sys.version}")
        return False
    
    # Check critical imports
    critical_modules = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("onnxruntime", "ONNX Runtime")
    ]
    
    missing_modules = []
    for module, description in critical_modules:
        try:
            __import__(module)
            logger.info(f"✅ {description}")
        except ImportError:
            logger.error(f"❌ {description} ({module}) not found")
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing modules: {', '.join(missing_modules)}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    logger = logging.getLogger(__name__)
    
    model_files = {
        "YOLOv9c": "model/face-detection/yolov9c-face-lindevs.onnx",
        "YOLOv9e": "model/face-detection/yolov9e-face-lindevs.onnx",
        "YOLOv11m": "model/face-detection/yolov11m-face.pt",
        "FaceNet": "model/face-recognition/facenet_vggface2.onnx",
        "AdaFace": "model/face-recognition/adaface_ir101.onnx", 
        "ArcFace": "model/face-recognition/arcface_r100.onnx"
    }
    
    available_models = []
    missing_models = []
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✅ {model_name}: {size_mb:.1f}MB")
            available_models.append(model_name)
        else:
            logger.warning(f"⚠️ {model_name}: Not found")
            missing_models.append(model_name)
    
    logger.info(f"Available models: {len(available_models)}/{len(model_files)}")
    
    if not available_models:
        logger.error("❌ No model files found! System cannot function.")
        return False
    
    return True

def check_gpu_availability():
    """Check GPU availability"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"✅ GPU {i}: {device_name} ({memory:.1f}GB)")
            return True
        else:
            logger.warning("⚠️ CUDA not available - using CPU mode")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Error checking GPU: {e}")
        return False

async def test_services():
    """Test service initialization"""
    logger = logging.getLogger(__name__)
    
    try:
        from src.core.config import get_settings
        from src.ai_services.common.vram_manager import VRAMManager
        
        logger.info("🔧 Testing service initialization...")
        
        # Test VRAM Manager
        settings = get_settings()
        vram_manager = VRAMManager(settings.vram_config)
        logger.info("✅ VRAM Manager initialized")
        
        # Test basic functionality
        vram_status = await vram_manager.get_vram_status()
        logger.info(f"✅ VRAM Status: {vram_status['total_vram'] / (1024*1024):.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Service test failed: {e}")
        return False

def start_server(host="0.0.0.0", port=8080, reload=True, workers=1):
    """Start the FastAPI server"""
    logger = logging.getLogger(__name__)
    
    try:
        import uvicorn
        
        logger.info("🚀 Starting Face Recognition System...")
        logger.info(f"   Host: {host}")
        logger.info(f"   Port: {port}")
        logger.info(f"   Reload: {reload}")
        logger.info(f"   Workers: {workers}")
        
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False

def main():
    """Main entry point"""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Face Recognition System Starter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--skip-checks", action="store_true", help="Skip system checks")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print("🎭 Face Recognition System Starter")
    print("=" * 50)
    
    if not args.skip_checks:
        # System checks
        logger.info("🔍 Checking system requirements...")
        if not check_system_requirements():
            logger.error("❌ System requirements not met")
            return False
        
        logger.info("🤖 Checking model files...")
        if not check_model_files():
            logger.error("❌ Model files check failed")
            return False
        
        logger.info("🔥 Checking GPU availability...")
        check_gpu_availability()
        
        logger.info("🧪 Testing services...")
        if not asyncio.run(test_services()):
            logger.error("❌ Service test failed")
            return False
        
        logger.info("✅ All checks passed!")
    
    # Start server
    start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        workers=args.workers
    )
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
