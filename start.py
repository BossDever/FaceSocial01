#!/usr/bin/env python3
"""
Start script for Face Recognition System
Fixed version with proper error handling and reload exclusions
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/startup.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def check_system_requirements() -> bool:
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

def check_model_files() -> int:
    """Check if model files exist and return count"""
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
            try:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                logger.info(f"✅ {model_name}: {size_mb:.1f}MB")
                available_models.append(model_name)
            except OSError as e:
                logger.warning(f"⚠️ {model_name}: Error reading file - {e}")
                missing_models.append(model_name)
        else:
            logger.warning(f"⚠️ {model_name}: Not found")
            missing_models.append(model_name)
    
    logger.info(f"Available models: {len(available_models)}/{len(model_files)}")
    
    if not available_models:
        logger.error("❌ No model files found! System will have limited functionality.")
        return 0
    
    return len(available_models)

def check_gpu_availability() -> bool:
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

async def test_basic_imports() -> bool:
    """Test basic service imports without full initialization"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔧 Testing basic imports...")
        
        # Test core config
        from src.core.config import get_settings
        settings = get_settings()
        logger.info("✅ Core config loaded")
        
        # Test VRAM Manager import
        from src.ai_services.common.vram_manager import VRAMManager
        logger.info("✅ VRAM Manager imported")
        
        # Test service imports
        from src.ai_services.face_detection.face_detection_service import FaceDetectionService
        from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService
        from src.ai_services.face_analysis.face_analysis_service import FaceAnalysisService
        logger.info("✅ All services imported")
        
        # Test API imports
        from src.api.complete_endpoints import face_detection_router, face_recognition_router, face_analysis_router
        logger.info("✅ API endpoints imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        logger.error("This indicates issues with the code structure or dependencies")
        return False

def create_required_directories() -> None:
    """Create required directories"""
    logger = logging.getLogger(__name__)
    
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
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"✅ Directory: {directory}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create {directory}: {e}")

def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = True,
    workers: int = 1,
    log_level: str = "info"
) -> bool:
    """Start the FastAPI server with proper reload configuration"""
    logger = logging.getLogger(__name__)
    
    try:
        import uvicorn
        
        logger.info("🚀 Starting Face Recognition System...")
        logger.info(f"   Host: {host}")
        logger.info(f"   Port: {port}")
        logger.info(f"   Reload: {reload}")
        logger.info(f"   Workers: {workers}")
        logger.info(f"   Log Level: {log_level}")
        
        # Define reload configuration to exclude problematic files/directories
        reload_dirs_config = [
            str(project_root / "src"),
            str(project_root / "config") if (project_root / "config").exists() else None,
        ]
        # Remove None values
        reload_dirs_config = [d for d in reload_dirs_config if d is not None]
        
        # Exclude patterns that cause reload loops
        reload_excludes_config = [
            "**/temp/**",
            "**/logs/**", 
            "**/output/**",
            "**/.mypy_cache/**",
            "**/__pycache__/**",
            "**/*.log",
            "**/*.tmp",
            "**/*.temp",
            "**/model/**",  # Exclude model files as they're large and don't need watching
        ]
        
        # Only watch specific file types
        reload_includes_config = [
            "*.py",
            "*.yml",
            "*.yaml", 
            "*.json",
            "*.toml",
        ]

        if reload:
            logger.info("🔄 Reload configuration:")
            logger.info(f"  Reload Dirs: {reload_dirs_config}")
            logger.info(f"  Reload Excludes: {reload_excludes_config}")
            logger.info(f"  Reload Includes: {reload_includes_config}")
        
        # Start server with appropriate configuration
        if reload:
            uvicorn.run(
                "src.main:app",
                host=host,
                port=port,
                reload=True,
                reload_dirs=reload_dirs_config,
                reload_excludes=reload_excludes_config,
                reload_includes=reload_includes_config,
                workers=1,  # Force single worker in reload mode
                log_level=log_level.lower(),
                access_log=True,
                use_colors=True,
            )
        else:
            uvicorn.run(
                "src.main:app",
                host=host,
                port=port,
                reload=False,
                workers=workers,
                log_level=log_level.lower(),
                access_log=True,
                use_colors=True
            )
        return True
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
        return True
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False

def print_startup_info(host: str, port: int) -> None:
    """Print startup information"""
    print("\n" + "="*60)
    print("🎭 Face Recognition System")
    print("="*60)
    print(f"🌐 Web Interface:      http://{host}:{port}")
    print(f"📚 API Documentation:  http://{host}:{port}/docs")
    print(f"📖 Alternative Docs:   http://{host}:{port}/redoc")
    print(f"🏥 Health Check:       http://{host}:{port}/health")
    print("="*60)
    print("💡 Press Ctrl+C to stop the server")
    print("="*60)

def main() -> bool:
    """Main entry point"""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Face Recognition System Starter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--skip-checks", action="store_true", help="Skip system checks")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    print("🎭 Face Recognition System Starter")
    print("=" * 50)
    
    # Create required directories
    create_required_directories()
    
    if not args.skip_checks:
        # System checks
        logger.info("🔍 Checking system requirements...")
        if not check_system_requirements():
            logger.error("❌ System requirements not met")
            return False
        
        logger.info("🤖 Checking model files...")
        model_count = check_model_files()
        if model_count == 0:
            logger.warning("⚠️ No model files found - system will have limited functionality")
        
        logger.info("🔥 Checking GPU availability...")
        check_gpu_availability()
        
        logger.info("🧪 Testing basic imports...")
        if not asyncio.run(test_basic_imports()):
            logger.error("❌ Import test failed")
            return False
        
        logger.info("✅ All checks passed!")
    else:
        logger.info("⏭️ Skipping system checks as requested")
    
    # Print startup information
    print_startup_info(args.host, args.port)
    
    # Start server
    return start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)