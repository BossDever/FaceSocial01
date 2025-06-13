"""
Face Recognition Service Package
Advanced face recognition using FaceNet, AdaFace, and ArcFace models
with comprehensive embedding extraction, matching, and gallery management
"""

import logging
from typing import Dict, Any, Optional, List, Union
import os

# Package metadata
__version__ = "2.0.0"
__package_name__ = "ai_services.face_recognition"

# Setup logging
logger = logging.getLogger(__name__)

# Supported recognition models
SUPPORTED_MODELS = {
    "facenet": {
        "name": "FaceNet VGGFace2",
        "file": "facenet_vggface2.onnx",
        "embedding_size": 512,
        "input_size": (160, 160),
        "description": "Robust general-purpose face recognition",
        "recommended_for": "general",
        "accuracy": "high",
        "speed": "fast"
    },
    "adaface": {
        "name": "AdaFace IR101", 
        "file": "adaface_ir101.onnx",
        "embedding_size": 512,
        "input_size": (112, 112),
        "description": "State-of-the-art accuracy with adaptive training",
        "recommended_for": "accuracy",
        "accuracy": "very_high", 
        "speed": "medium"
    },
    "arcface": {
        "name": "ArcFace R100",
        "file": "arcface_r100.onnx", 
        "embedding_size": 512,
        "input_size": (112, 112),
        "description": "Strong feature learning with angular margin",
        "recommended_for": "robustness",
        "accuracy": "very_high",
        "speed": "medium"
    }
}

# Default similarity thresholds for each model (Optimized for real-world performance)
DEFAULT_THRESHOLDS = {
    "facenet": {
        "similarity_threshold": 0.50,  # Lowered from 0.60 to 0.50
        "unknown_threshold": 0.45,     # Lowered from 0.55 to 0.45
        "high_confidence": 0.75        # Lowered from 0.80 to 0.75
    },
    "adaface": {
        "similarity_threshold": 0.55,  # Lowered from 0.65 to 0.55
        "unknown_threshold": 0.50,     # Lowered from 0.60 to 0.50
        "high_confidence": 0.80        # Lowered from 0.85 to 0.80
    },
    "arcface": {
        "similarity_threshold": 0.55,  # Lowered from 0.65 to 0.55
        "unknown_threshold": 0.50,     # Lowered from 0.60 to 0.50
        "high_confidence": 0.80        # Lowered from 0.85 to 0.80
    }
}

def get_face_recognition_service():
    """Lazy import for Face Recognition Service."""
    try:
        from .face_recognition_service import FaceRecognitionService
        return FaceRecognitionService
    except ImportError as e:
        logger.error(f"Could not import FaceRecognitionService: {e}")
        return None

def get_recognition_models():
    """Lazy import for recognition models and data structures."""
    try:
        from .models import (
            RecognitionModel,
            FaceEmbedding,
            FaceMatch,
            FaceComparisonResult,
            FaceRecognitionResult,
            RecognitionConfig,
            FaceQuality,
            RecognitionQuality
        )
        return {
            'RecognitionModel': RecognitionModel,
            'FaceEmbedding': FaceEmbedding,
            'FaceMatch': FaceMatch,
            'FaceComparisonResult': FaceComparisonResult,
            'FaceRecognitionResult': FaceRecognitionResult,
            'RecognitionConfig': RecognitionConfig,
            'FaceQuality': FaceQuality,
            'RecognitionQuality': RecognitionQuality
        }
    except ImportError as e:
        logger.error(f"Could not import recognition models: {e}")
        return None

def check_model_availability() -> Dict[str, bool]:
    """
    Check availability of recognition models.
    
    Returns:
        Dictionary with model availability status
    """
    availability = {}
    model_base_path = "model/face-recognition"
    
    for model_id, model_info in SUPPORTED_MODELS.items():
        model_file = model_info.get("file")
        if model_file:
            model_path = os.path.join(model_base_path, model_file)
            availability[model_id] = os.path.exists(model_path)
        else:
            availability[model_id] = False
    
    return availability

def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information dictionary or None if not found
    """
    model_info = SUPPORTED_MODELS.get(model_name)
    if model_info:
        # Add availability and threshold information
        availability = check_model_availability()
        thresholds = DEFAULT_THRESHOLDS.get(model_name, {})
        
        return {
            **model_info,
            "available": availability.get(model_name, False),
            "thresholds": thresholds,
            "model_path": os.path.join("model/face-recognition", model_info["file"])
        }
    return None

def get_recommended_model(priority: str = "general") -> Optional[str]:
    """
    Get recommended model based on priority.
    
    Args:
        priority: "general", "accuracy", "speed", "robustness"
        
    Returns:
        Model name or None if no suitable model available
    """
    availability = check_model_availability()
    available_models = [model for model, available in availability.items() if available]
    
    if not available_models:
        return None
    
    # Priority-based recommendations
    recommendations = {
        "general": ["facenet", "adaface", "arcface"],
        "accuracy": ["adaface", "arcface", "facenet"], 
        "speed": ["facenet", "adaface", "arcface"],
        "robustness": ["arcface", "adaface", "facenet"]
    }
    
    preferred_order = recommendations.get(priority, recommendations["general"])
    
    # Return first available model from preferred order
    for model in preferred_order:
        if model in available_models:
            return model
    
    # Fallback to any available model
    return available_models[0] if available_models else None

def create_recognition_service(vram_manager: Any = None, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    Factory function to create a face recognition service.
    
    Args:
        vram_manager: VRAM manager instance
        config: Service configuration
        
    Returns:
        FaceRecognitionService instance or None if creation failed
    """
    service_class = get_face_recognition_service()
    if service_class is None:
        logger.error("FaceRecognitionService not available")
        return None
    
    try:
        service = service_class(vram_manager=vram_manager, config=config)
        logger.info("Face recognition service created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to create face recognition service: {e}")
        return None

def get_default_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default configuration for face recognition.
    
    Args:
        model_name: Specific model name for model-specific defaults
        
    Returns:
        Default configuration dictionary
    """
    # Use recommended model if none specified
    if model_name is None:
        model_name = get_recommended_model("general")
    
    # Get model-specific thresholds
    thresholds = DEFAULT_THRESHOLDS.get(model_name, DEFAULT_THRESHOLDS["facenet"])
    
    return {
        "preferred_model": model_name or "facenet",
        "similarity_threshold": thresholds["similarity_threshold"],
        "unknown_threshold": thresholds["unknown_threshold"],
        "embedding_dimension": 512,
        "enable_gpu_optimization": True,
        "batch_size": 8,
        "quality_threshold": 0.2,
        "cuda_memory_fraction": 0.8,
        "use_cuda_graphs": False,
        "parallel_processing": True,
        "enable_quality_assessment": True,
        "auto_model_selection": True,
        "enable_unknown_detection": True,
        "max_faces": 10
    }

def validate_recognition_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize recognition configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
    """
    default_config = get_default_config()
    
    # Merge with defaults
    validated_config = default_config.copy()
    validated_config.update(config)
    
    # Validate model name
    model_name = validated_config.get("preferred_model", "facenet")
    if model_name not in SUPPORTED_MODELS:
        logger.warning(f"Unknown model '{model_name}', using 'facenet'")
        validated_config["preferred_model"] = "facenet"
    
    # Validate thresholds
    validated_config["similarity_threshold"] = max(0.1, min(1.0, 
        validated_config.get("similarity_threshold", 0.60)))
    validated_config["unknown_threshold"] = max(0.1, min(1.0,
        validated_config.get("unknown_threshold", 0.55)))
    validated_config["quality_threshold"] = max(0.0, min(1.0,
        validated_config.get("quality_threshold", 0.2)))
    
    # Validate sizes
    validated_config["batch_size"] = max(1, min(32, 
        validated_config.get("batch_size", 8)))
    validated_config["max_faces"] = max(1, min(100,
        validated_config.get("max_faces", 10)))
    validated_config["embedding_dimension"] = max(128, min(2048,
        validated_config.get("embedding_dimension", 512)))
    
    return validated_config

def create_empty_gallery() -> Dict[str, Any]:
    """
    Create an empty gallery structure.
    
    Returns:
        Empty gallery dictionary
    """
    return {}

def validate_gallery_format(gallery: Dict[str, Any]) -> bool:
    """
    Validate gallery format.
    
    Args:
        gallery: Gallery to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(gallery, dict):
        return False
    
    for person_id, person_data in gallery.items():
        if not isinstance(person_data, dict):
            return False
        
        # Check required fields
        if "name" not in person_data or "embeddings" not in person_data:
            return False
        
        # Check embeddings format
        embeddings = person_data["embeddings"]
        if not isinstance(embeddings, list):
            return False
        
        # Check each embedding
        for embedding in embeddings:
            if isinstance(embedding, dict):
                if "embedding" not in embedding:
                    return False
            elif not isinstance(embedding, (list, tuple)):
                return False
    
    return True

def get_package_capabilities() -> Dict[str, Any]:
    """
    Get package capabilities and status.
    
    Returns:
        Dictionary with capability information
    """
    capabilities = {
        "version": __version__,
        "supported_models": list(SUPPORTED_MODELS.keys()),
        "model_availability": check_model_availability(),
        "default_thresholds": DEFAULT_THRESHOLDS,
        "module_status": {}
    }
    
    # Check module availability
    modules = {
        "face_recognition_service": get_face_recognition_service() is not None,
        "recognition_models": get_recognition_models() is not None
    }
    
    capabilities["module_status"] = modules
    capabilities["fully_functional"] = all(modules.values())
    
    # Add recommended models
    capabilities["recommendations"] = {
        "general": get_recommended_model("general"),
        "accuracy": get_recommended_model("accuracy"), 
        "speed": get_recommended_model("speed"),
        "robustness": get_recommended_model("robustness")
    }
    
    return capabilities

def get_embeddings_info() -> Dict[str, Any]:
    """
    Get information about embedding formats and sizes.
    
    Returns:
        Embedding information dictionary
    """
    return {
        "standard_dimension": 512,
        "supported_dimensions": [128, 256, 512, 1024],
        "vector_format": "float32",
        "normalization": "L2 normalized",
        "similarity_metric": "cosine_similarity",
        "supported_formats": ["numpy_array", "list", "bytes"]
    }

# Export public interface
__all__ = [
    '__version__',
    '__package_name__',
    'SUPPORTED_MODELS', 
    'DEFAULT_THRESHOLDS',
    'get_face_recognition_service',
    'get_recognition_models',
    'check_model_availability',
    'get_model_info',
    'get_recommended_model',
    'create_recognition_service',
    'get_default_config',
    'validate_recognition_config',
    'create_empty_gallery',
    'validate_gallery_format',
    'get_package_capabilities',
    'get_embeddings_info'
]

# Package initialization
logger.info(f"Face Recognition package v{__version__} loaded")

# Check package capabilities on import
try:
    capabilities = get_package_capabilities()
    available_models = sum(capabilities["model_availability"].values())
    total_models = len(capabilities["supported_models"])
    
    logger.info(f"Package capabilities: {available_models}/{total_models} models available")
    
    if not capabilities["fully_functional"]:
        missing_modules = [name for name, available in capabilities["module_status"].items() if not available]
        logger.warning(f"Missing modules: {missing_modules}")
        
    if available_models == 0:
        logger.warning("No recognition models available - please install model files")
    else:
        recommended = capabilities["recommendations"]["general"]
        if recommended:
            logger.info(f"Recommended model for general use: {recommended}")
        
except Exception as e:
    logger.warning(f"Failed to check package capabilities: {e}")