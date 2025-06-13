"""
Face Detection Service Package
Advanced face detection using YOLOv9c, YOLOv9e, and YOLOv11m models
with intelligent fallback system and quality assessment
"""

import logging
from typing import Dict, Any, Optional, List, Union

# Package metadata
__version__ = "2.0.0"
__package_name__ = "ai_services.face_detection"

# Setup logging
logger = logging.getLogger(__name__)

# Available detection models
SUPPORTED_MODELS = {
    "yolov9c": {
        "name": "YOLOv9c Face Detection",
        "file": "yolov9c-face-lindevs.onnx",
        "type": "onnx",
        "description": "Fast and efficient face detection",
        "recommended_for": "speed"
    },
    "yolov9e": {
        "name": "YOLOv9e Face Detection", 
        "file": "yolov9e-face-lindevs.onnx",
        "type": "onnx",
        "description": "High accuracy face detection",
        "recommended_for": "accuracy"
    },
    "yolov11m": {
        "name": "YOLOv11m Face Detection",
        "file": "yolov11m-face.pt", 
        "type": "pytorch",
        "description": "Balanced speed and accuracy",
        "recommended_for": "balanced"
    },
    "auto": {
        "name": "Auto Selection",
        "description": "Automatically select best available model",
        "recommended_for": "default"
    }
}

# Fallback detection methods
FALLBACK_METHODS = [
    "opencv_haar",  # OpenCV Haar Cascade
    "dlib_hog",     # Dlib HOG (if available) 
    "mtcnn"         # MTCNN (if available)
]

def get_face_detection_service():
    """Lazy import for Face Detection Service."""
    try:
        from .face_detection_service import FaceDetectionService
        return FaceDetectionService
    except ImportError as e:
        logger.error(f"Could not import FaceDetectionService: {e}")
        return None

def get_yolo_models():
    """Lazy import for YOLO models."""
    try:
        from .yolo_models import (
            FaceDetector,
            YOLOv9ONNXDetector,
            YOLOv11Detector,
            fallback_opencv_detection
        )
        return {
            'FaceDetector': FaceDetector,
            'YOLOv9ONNXDetector': YOLOv9ONNXDetector,
            'YOLOv11Detector': YOLOv11Detector,
            'fallback_opencv_detection': fallback_opencv_detection
        }
    except ImportError as e:
        logger.error(f"Could not import YOLO models: {e}")
        return None

def get_detection_utils():
    """Lazy import for detection utilities."""
    try:
        from .utils import (
            BoundingBox,
            FaceDetection,
            DetectionResult,
            FaceQualityAnalyzer,
            calculate_face_quality,
            validate_bounding_box,
            filter_detection_results,
            draw_detection_results,
            save_detection_image,
            get_relaxed_face_detection_config
        )
        return {
            'BoundingBox': BoundingBox,
            'FaceDetection': FaceDetection,
            'DetectionResult': DetectionResult,
            'FaceQualityAnalyzer': FaceQualityAnalyzer,
            'calculate_face_quality': calculate_face_quality,
            'validate_bounding_box': validate_bounding_box,
            'filter_detection_results': filter_detection_results,
            'draw_detection_results': draw_detection_results,
            'save_detection_image': save_detection_image,
            'get_relaxed_face_detection_config': get_relaxed_face_detection_config
        }
    except ImportError as e:
        logger.error(f"Could not import detection utilities: {e}")
        return None

def check_model_availability() -> Dict[str, bool]:
    """
    Check availability of detection models.
    
    Returns:
        Dictionary with model availability status
    """
    import os
    availability = {}
    
    model_base_path = "model/face-detection"
    
    for model_id, model_info in SUPPORTED_MODELS.items():
        if model_id == "auto":
            # Auto is always available if any other model is available
            continue
            
        model_file = model_info.get("file")
        if model_file:
            model_path = os.path.join(model_base_path, model_file)
            availability[model_id] = os.path.exists(model_path)
        else:
            availability[model_id] = False
    
    # Set auto availability based on other models
    availability["auto"] = any(availability.values())
    
    return availability

def get_recommended_model(priority: str = "balanced") -> Optional[str]:
    """
    Get recommended model based on priority.
    
    Args:
        priority: "speed", "accuracy", "balanced", or "auto"
        
    Returns:
        Model name or None if no suitable model available
    """
    availability = check_model_availability()
    
    if priority == "auto" or priority == "balanced":
        # Prefer YOLOv11m for balanced performance
        if availability.get("yolov11m"):
            return "yolov11m"
        elif availability.get("yolov9c"):
            return "yolov9c"
        elif availability.get("yolov9e"):
            return "yolov9e"
            
    elif priority == "speed":
        # Prefer YOLOv9c for speed
        if availability.get("yolov9c"):
            return "yolov9c"
        elif availability.get("yolov11m"):
            return "yolov11m"
        elif availability.get("yolov9e"):
            return "yolov9e"
            
    elif priority == "accuracy":
        # Prefer YOLOv9e for accuracy
        if availability.get("yolov9e"):
            return "yolov9e"
        elif availability.get("yolov11m"):
            return "yolov11m"
        elif availability.get("yolov9c"):
            return "yolov9c"
    
    return None

def create_detection_service(vram_manager: Any = None, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    Factory function to create a face detection service.
    
    Args:
        vram_manager: VRAM manager instance
        config: Service configuration
        
    Returns:
        FaceDetectionService instance or None if creation failed
    """
    service_class = get_face_detection_service()
    if service_class is None:
        logger.error("FaceDetectionService not available")
        return None
    
    try:
        service = service_class(vram_manager=vram_manager, config=config)
        logger.info("Face detection service created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to create face detection service: {e}")
        return None

def get_package_capabilities() -> Dict[str, Any]:
    """
    Get package capabilities and status.
    
    Returns:
        Dictionary with capability information
    """
    capabilities = {
        "version": __version__,
        "supported_models": list(SUPPORTED_MODELS.keys()),
        "fallback_methods": FALLBACK_METHODS,
        "model_availability": check_model_availability(),
        "module_status": {}
    }
    
    # Check module availability
    modules = {
        "face_detection_service": get_face_detection_service() is not None,
        "yolo_models": get_yolo_models() is not None,
        "detection_utils": get_detection_utils() is not None
    }
    
    capabilities["module_status"] = modules
    capabilities["fully_functional"] = all(modules.values())
    
    return capabilities

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for face detection.
    
    Returns:
        Default configuration dictionary
    """
    utils = get_detection_utils()
    if utils and 'get_relaxed_face_detection_config' in utils:
        return utils['get_relaxed_face_detection_config']()
    else:
        # Fallback configuration if utils not available
        return {
            "use_enhanced_detector": False,
            "conf_threshold": 0.10,
            "iou_threshold_nms": 0.35,
            "min_quality_threshold": 40,
            "max_faces": 50,
            "enable_fallback_system": True,
            "fallback_models": [
                {
                    "model_name": "yolov11m",
                    "conf_threshold": 0.15,
                    "min_faces_to_accept": 1
                },
                {
                    "model_name": "opencv_haar",
                    "scale_factor": 1.1,
                    "min_neighbors": 3,
                    "min_size": (20, 20),
                    "min_faces_to_accept": 1
                }
            ]
        }

def validate_detection_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize detection configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
    """
    default_config = get_default_config()
    
    # Merge with defaults
    validated_config = default_config.copy()
    validated_config.update(config)
    
    # Validate ranges
    validated_config["conf_threshold"] = max(0.01, min(1.0, validated_config.get("conf_threshold", 0.10)))
    validated_config["iou_threshold_nms"] = max(0.01, min(1.0, validated_config.get("iou_threshold_nms", 0.35)))
    validated_config["min_quality_threshold"] = max(0, min(100, validated_config.get("min_quality_threshold", 40)))
    validated_config["max_faces"] = max(1, min(100, validated_config.get("max_faces", 50)))
    
    return validated_config

# Export public interface
__all__ = [
    '__version__',
    '__package_name__',
    'SUPPORTED_MODELS',
    'FALLBACK_METHODS',
    'get_face_detection_service',
    'get_yolo_models',
    'get_detection_utils',
    'check_model_availability',
    'get_recommended_model',
    'create_detection_service',
    'get_package_capabilities',
    'get_default_config',
    'validate_detection_config'
]

# Package initialization
logger.info(f"Face Detection package v{__version__} loaded")

# Check package capabilities on import
try:
    capabilities = get_package_capabilities()
    available_models = sum(capabilities["model_availability"].values())
    total_models = len([m for m in capabilities["supported_models"] if m != "auto"])
    
    logger.info(f"Package capabilities: {available_models}/{total_models} models available")
    
    if not capabilities["fully_functional"]:
        missing_modules = [name for name, available in capabilities["module_status"].items() if not available]
        logger.warning(f"Missing modules: {missing_modules}")
        
    if available_models == 0:
        logger.warning("No detection models available - please install model files")
        
except Exception as e:
    logger.warning(f"Failed to check package capabilities: {e}")