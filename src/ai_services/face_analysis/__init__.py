"""
Face Analysis Service Package
Comprehensive face analysis combining detection and recognition
with advanced analytics, batch processing, and quality assessment
"""

import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum

# Package metadata
__version__ = "2.0.0"
__package_name__ = "ai_services.face_analysis"

# Setup logging
logger = logging.getLogger(__name__)

# Analysis modes available
class AnalysisModeEnum(Enum):
    """Available analysis modes"""
    DETECTION_ONLY = "detection_only"      # Only face detection
    RECOGNITION_ONLY = "recognition_only"  # Only face recognition  
    FULL_ANALYSIS = "full_analysis"        # Detection + Recognition
    COMPREHENSIVE = "comprehensive"        # Full analysis + advanced features
    VERIFICATION = "verification"          # Face verification (1:1 comparison)

# Quality levels for analysis
class QualityLevelEnum(Enum):
    """Quality levels for analysis processing"""
    HIGH = "high"        # High quality, slower processing
    BALANCED = "balanced" # Balanced quality and speed
    FAST = "fast"        # Fast processing, lower quality

# Analysis capabilities
ANALYSIS_CAPABILITIES = {
    "modes": {
        "detection_only": {
            "description": "Face detection only with quality assessment",
            "features": ["face_detection", "quality_scoring", "bounding_boxes"],
            "output": ["faces", "statistics"]
        },
        "recognition_only": {
            "description": "Face recognition without detection",
            "features": ["face_recognition", "gallery_matching", "embeddings"],
            "output": ["matches", "similarities", "embeddings"]
        },
        "full_analysis": {
            "description": "Complete detection and recognition pipeline",
            "features": ["face_detection", "face_recognition", "gallery_matching"],
            "output": ["faces", "matches", "statistics", "embeddings"]
        },
        "comprehensive": {
            "description": "Full analysis with advanced features",
            "features": ["face_detection", "face_recognition", "quality_assessment", 
                        "batch_processing", "analytics"],
            "output": ["faces", "matches", "statistics", "analytics", "embeddings"]
        },
        "verification": {
            "description": "1:1 face verification",
            "features": ["face_comparison", "similarity_scoring"],
            "output": ["similarity", "match_result", "confidence"]
        }
    },
    "quality_levels": {
        "high": {
            "description": "Maximum accuracy with detailed analysis",
            "processing_time": "slow",
            "accuracy": "highest",
            "features": ["enhanced_detection", "quality_filtering", "advanced_recognition"]
        },
        "balanced": {
            "description": "Good balance of speed and accuracy",
            "processing_time": "medium", 
            "accuracy": "high",
            "features": ["standard_detection", "basic_filtering", "standard_recognition"]
        },
        "fast": {
            "description": "Fast processing for real-time applications",
            "processing_time": "fast",
            "accuracy": "good", 
            "features": ["fast_detection", "minimal_filtering", "fast_recognition"]
        }
    }
}

def get_face_analysis_service():
    """Lazy import for Face Analysis Service."""
    try:
        from .face_analysis_service import FaceAnalysisService
        return FaceAnalysisService
    except ImportError as e:
        logger.error(f"Could not import FaceAnalysisService: {e}")
        return None

def get_analysis_models():
    """Lazy import for analysis models and data structures."""
    try:
        from .models import (
            AnalysisConfig,
            FaceResult,
            FaceAnalysisResult,
            BatchAnalysisResult,
            AnalysisMode,
            QualityLevel,
            DetectionConfig,
            FaceAnalysisJSONRequest
        )
        return {
            'AnalysisConfig': AnalysisConfig,
            'FaceResult': FaceResult, 
            'FaceAnalysisResult': FaceAnalysisResult,
            'BatchAnalysisResult': BatchAnalysisResult,
            'AnalysisMode': AnalysisMode,
            'QualityLevel': QualityLevel,
            'DetectionConfig': DetectionConfig,
            'FaceAnalysisJSONRequest': FaceAnalysisJSONRequest
        }
    except ImportError as e:
        logger.error(f"Could not import analysis models: {e}")
        return None

def create_analysis_service(
    vram_manager: Any = None, 
    config: Optional[Dict[str, Any]] = None,
    face_detection_service: Any = None,
    face_recognition_service: Any = None
) -> Optional[Any]:
    """
    Factory function to create a face analysis service.
    
    Args:
        vram_manager: VRAM manager instance
        config: Service configuration
        face_detection_service: Pre-initialized face detection service
        face_recognition_service: Pre-initialized face recognition service
        
    Returns:
        FaceAnalysisService instance or None if creation failed
    """
    service_class = get_face_analysis_service()
    if service_class is None:
        logger.error("FaceAnalysisService not available")
        return None
    
    try:
        service = service_class(
            vram_manager=vram_manager,
            config=config,
            face_detection_service=face_detection_service,
            face_recognition_service=face_recognition_service
        )
        logger.info("Face analysis service created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to create face analysis service: {e}")
        return None

def get_default_config(mode: str = "full_analysis", quality_level: str = "balanced") -> Dict[str, Any]:
    """
    Get default configuration for face analysis.
    
    Args:
        mode: Analysis mode
        quality_level: Quality level for processing
        
    Returns:
        Default configuration dictionary
    """
    # Quality-based settings
    quality_configs = {
        "high": {
            "confidence_threshold": 0.3,
            "min_face_size": 40,
            "min_quality_threshold": 70,
            "parallel_processing": False,
            "use_quality_based_selection": True
        },
        "balanced": {
            "confidence_threshold": 0.5,
            "min_face_size": 32,
            "min_quality_threshold": 50,
            "parallel_processing": True,
            "use_quality_based_selection": True
        },
        "fast": {
            "confidence_threshold": 0.7,
            "min_face_size": 24,
            "min_quality_threshold": 30,
            "parallel_processing": True,
            "use_quality_based_selection": False
        }
    }
    
    # Get quality-specific settings
    quality_settings = quality_configs.get(quality_level, quality_configs["balanced"])
    
    base_config = {
        "mode": mode,
        "quality_level": quality_level,
        
        # Detection settings
        "detection_model": "auto",
        "confidence_threshold": quality_settings["confidence_threshold"],
        "min_face_size": quality_settings["min_face_size"],
        "max_faces": 50,
        
        # Recognition settings
        "recognition_model": "facenet",
        "enable_embedding_extraction": True,
        "enable_gallery_matching": True,
        "enable_database_matching": True,
        "gallery_top_k": 5,
        
        # Performance settings
        "batch_size": 8,
        "parallel_processing": quality_settings["parallel_processing"],
        "use_quality_based_selection": quality_settings["use_quality_based_selection"],
        
        # Output settings
        "return_face_crops": False,
        "return_embeddings": False,
        "return_detailed_stats": True,
        "recognition_image_format": "jpg"
    }
    
    return base_config

def validate_analysis_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize analysis configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
    """
    default_config = get_default_config()
    
    # Merge with defaults
    validated_config = default_config.copy()
    validated_config.update(config)
    
    # Validate mode
    valid_modes = [mode.value for mode in AnalysisModeEnum]
    if validated_config.get("mode") not in valid_modes:
        logger.warning(f"Invalid mode '{validated_config.get('mode')}', using 'full_analysis'")
        validated_config["mode"] = "full_analysis"
    
    # Validate quality level
    valid_quality_levels = [level.value for level in QualityLevelEnum]
    if validated_config.get("quality_level") not in valid_quality_levels:
        logger.warning(f"Invalid quality level '{validated_config.get('quality_level')}', using 'balanced'")
        validated_config["quality_level"] = "balanced"
    
    # Validate numeric parameters
    validated_config["confidence_threshold"] = max(0.01, min(1.0, 
        validated_config.get("confidence_threshold", 0.5)))
    validated_config["min_face_size"] = max(8, min(200,
        validated_config.get("min_face_size", 32)))
    validated_config["max_faces"] = max(1, min(100,
        validated_config.get("max_faces", 50)))
    validated_config["batch_size"] = max(1, min(32,
        validated_config.get("batch_size", 8)))
    validated_config["gallery_top_k"] = max(1, min(20,
        validated_config.get("gallery_top_k", 5)))
    
    return validated_config

def create_analysis_config(
    mode: str = "full_analysis",
    detection_model: str = "auto", 
    recognition_model: str = "facenet",
    quality_level: str = "balanced",
    **kwargs
) -> Dict[str, Any]:
    """
    Create analysis configuration with common parameters.
    
    Args:
        mode: Analysis mode
        detection_model: Detection model to use
        recognition_model: Recognition model to use
        quality_level: Quality level
        **kwargs: Additional configuration parameters
        
    Returns:
        Analysis configuration dictionary
    """
    config = get_default_config(mode, quality_level)
    
    # Override specific parameters
    config.update({
        "detection_model": detection_model,
        "recognition_model": recognition_model,
        **kwargs
    })
    
    return validate_analysis_config(config)

def get_supported_formats() -> Dict[str, List[str]]:
    """
    Get supported input and output formats.
    
    Returns:
        Dictionary with supported formats
    """
    return {
        "input_formats": {
            "image_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            "input_types": ["file_upload", "base64_string", "numpy_array", "url"],
            "batch_types": ["multiple_files", "archive", "directory"]
        },
        "output_formats": {
            "result_formats": ["json", "xml", "csv"],
            "image_formats": ["jpg", "png", "pdf"],
            "data_formats": ["numpy", "list", "dataframe"]
        },
        "gallery_formats": {
            "input": ["json", "csv", "database"],
            "storage": ["memory", "file", "database"]
        }
    }

def get_performance_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Get performance recommendations for different use cases.
    
    Returns:
        Performance recommendations dictionary
    """
    return {
        "real_time": {
            "quality_level": "fast",
            "max_faces": 10,
            "parallel_processing": True,
            "detection_model": "yolov9c",
            "recognition_model": "facenet"
        },
        "batch_processing": {
            "quality_level": "balanced", 
            "max_faces": 50,
            "parallel_processing": True,
            "detection_model": "auto",
            "recognition_model": "adaface"
        },
        "high_accuracy": {
            "quality_level": "high",
            "max_faces": 20,
            "parallel_processing": False,
            "detection_model": "yolov9e", 
            "recognition_model": "arcface"
        },
        "security_application": {
            "quality_level": "high",
            "max_faces": 10,
            "parallel_processing": False,
            "detection_model": "yolov9e",
            "recognition_model": "arcface",
            "use_quality_based_selection": True
        }
    }

def get_package_capabilities() -> Dict[str, Any]:
    """
    Get package capabilities and status.
    
    Returns:
        Dictionary with capability information
    """
    capabilities = {
        "version": __version__,
        "analysis_modes": list(AnalysisModeEnum),
        "quality_levels": list(QualityLevelEnum),
        "capabilities": ANALYSIS_CAPABILITIES,
        "supported_formats": get_supported_formats(),
        "performance_recommendations": get_performance_recommendations(),
        "module_status": {}
    }
    
    # Check module availability
    modules = {
        "face_analysis_service": get_face_analysis_service() is not None,
        "analysis_models": get_analysis_models() is not None
    }
    
    capabilities["module_status"] = modules
    capabilities["fully_functional"] = all(modules.values())
    
    return capabilities

def get_analysis_statistics_template() -> Dict[str, Any]:
    """
    Get template for analysis statistics.
    
    Returns:
        Statistics template dictionary
    """
    return {
        "total_faces": 0,
        "usable_faces": 0,
        "identified_faces": 0,
        "unique_identities": 0,
        "detection_success_rate": 0.0,
        "recognition_success_rate": 0.0,
        "average_detection_confidence": 0.0,
        "average_quality": 0.0,
        "average_recognition_confidence": 0.0,
        "processing_time": {
            "detection_time": 0.0,
            "recognition_time": 0.0,
            "total_time": 0.0
        },
        "quality_distribution": {
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "poor": 0
        }
    }

# Export public interface
__all__ = [
    '__version__',
    '__package_name__',
    'AnalysisModeEnum',
    'QualityLevelEnum', 
    'ANALYSIS_CAPABILITIES',
    'get_face_analysis_service',
    'get_analysis_models',
    'create_analysis_service',
    'get_default_config',
    'validate_analysis_config',
    'create_analysis_config',
    'get_supported_formats',
    'get_performance_recommendations',
    'get_package_capabilities',
    'get_analysis_statistics_template'
]

# Package initialization
logger.info(f"Face Analysis package v{__version__} loaded")

# Check package capabilities on import
try:
    capabilities = get_package_capabilities()
    
    logger.info(f"Package capabilities: {len(capabilities['analysis_modes'])} modes, "
               f"{len(capabilities['quality_levels'])} quality levels available")
    
    if not capabilities["fully_functional"]:
        missing_modules = [name for name, available in capabilities["module_status"].items() if not available]
        logger.warning(f"Missing modules: {missing_modules}")
    else:
        logger.info("All analysis modules available")
        
except Exception as e:
    logger.warning(f"Failed to check package capabilities: {e}")