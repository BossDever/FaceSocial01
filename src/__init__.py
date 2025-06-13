"""
Face Recognition System - Source Package
Professional Face Detection, Recognition & Analysis System
"""

__version__ = "2.0.0"
__author__ = "Face Recognition System Team"
__description__ = "Professional Face Detection, Recognition & Analysis System with GPU optimization"

# Package metadata
__title__ = "face-recognition-system"
__license__ = "MIT"
__copyright__ = "2024 Face Recognition System"

# Version info tuple
VERSION_INFO = (2, 0, 0)

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 8, 0)

# Check Python version compatibility
import sys
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"Face Recognition System requires Python {'.'.join(map(str, MIN_PYTHON_VERSION))} "
        f"or higher. Current version: {'.'.join(map(str, sys.version_info[:3]))}"
    )

# Core imports for package initialization
try:
    from .core.config import get_settings
    from .core.log_config import get_logger, setup_logging
    
    # Initialize logging
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"Face Recognition System v{__version__} initialized")
    
except ImportError as e:
    # Fallback for cases where dependencies might not be available
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import core modules: {e}")

# Export key components
__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "__title__",
    "__license__",
    "__copyright__",
    "VERSION_INFO",
    "MIN_PYTHON_VERSION",
    "get_settings",
    "get_logger",
    "setup_logging"
]

# Optional feature flags (can be set by environment variables)
import os

# Feature flags
ENABLE_GPU = os.getenv("FACE_RECOGNITION_GPU_ENABLED", "true").lower() in ("true", "1", "yes")
ENABLE_DEBUG = os.getenv("FACE_RECOGNITION_DEBUG", "false").lower() in ("true", "1", "yes")
ENABLE_METRICS = os.getenv("FACE_RECOGNITION_METRICS", "true").lower() in ("true", "1", "yes")

# Export feature flags
__all__.extend([
    "ENABLE_GPU",
    "ENABLE_DEBUG", 
    "ENABLE_METRICS"
])

def get_version() -> str:
    """Get the current version string."""
    return __version__

def get_version_info() -> tuple:
    """Get the current version info tuple."""
    return VERSION_INFO

def check_dependencies() -> dict:
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: Dictionary with dependency status
    """
    dependencies = {
        "fastapi": False,
        "uvicorn": False,
        "torch": False,
        "opencv": False,
        "numpy": False,
        "onnxruntime": False,
        "pydantic": False
    }
    
    # Check each dependency
    for dep in dependencies:
        try:
            if dep == "opencv":
                import cv2
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            pass
    
    return dependencies

def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        dict: System information
    """
    info = {
        "version": __version__,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "platform": sys.platform,
        "dependencies": check_dependencies(),
        "feature_flags": {
            "gpu_enabled": ENABLE_GPU,
            "debug_enabled": ENABLE_DEBUG,
            "metrics_enabled": ENABLE_METRICS
        }
    }
    
    # Add GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_info"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            }
        else:
            info["gpu_info"] = {"available": False}
    except ImportError:
        info["gpu_info"] = {"available": False, "reason": "PyTorch not installed"}
    
    return info

# Add system info to exports
__all__.extend([
    "get_version",
    "get_version_info", 
    "check_dependencies",
    "get_system_info"
])

# Initialize package-level logger
if 'logger' in locals():
    logger.debug(f"Package initialized with features: GPU={ENABLE_GPU}, Debug={ENABLE_DEBUG}, Metrics={ENABLE_METRICS}")