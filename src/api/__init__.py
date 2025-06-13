"""
API Package for Face Recognition System
FastAPI-based REST API endpoints for face detection, recognition, and analysis
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter

# Package metadata
__version__ = "2.0.0"
__package_name__ = "api"

# Setup logging
logger = logging.getLogger(__name__)

# API configuration
API_CONFIG = {
    "title": "Face Recognition API",
    "description": "Professional Face Detection, Recognition & Analysis System API",
    "version": __version__,
    "terms_of_service": None,
    "contact": {
        "name": "Face Recognition System",
        "email": "support@face-recognition-system.com"
    },
    "license_info": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
}

# API endpoint groups
API_GROUPS = {
    "face_detection": {
        "prefix": "/api/face-detection",
        "tags": ["Face Detection"],
        "description": "Face detection endpoints using YOLO models"
    },
    "face_recognition": {
        "prefix": "/api/face-recognition", 
        "tags": ["Face Recognition"],
        "description": "Face recognition and gallery management endpoints"
    },
    "face_analysis": {
        "prefix": "/api/face-analysis",
        "tags": ["Face Analysis"], 
        "description": "Comprehensive face analysis endpoints"
    },
    "system": {
        "prefix": "/api/system",
        "tags": ["System"],
        "description": "System health and monitoring endpoints"
    }
}

# Supported image formats and limits
IMAGE_CONFIG = {
    "supported_formats": ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff", "image/webp"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "max_batch_size": 20,  # Maximum files in batch upload
    "max_image_dimension": 4096,  # Maximum width/height
    "min_image_dimension": 32   # Minimum width/height
}

def get_complete_endpoints():
    """Lazy import for complete endpoints."""
    try:
        from .complete_endpoints import (
            face_detection_router,
            face_recognition_router, 
            face_analysis_router
        )
        return {
            'face_detection_router': face_detection_router,
            'face_recognition_router': face_recognition_router,
            'face_analysis_router': face_analysis_router
        }
    except ImportError as e:
        logger.error(f"Could not import complete endpoints: {e}")
        return None

def get_individual_endpoints():
    """Lazy import for individual endpoint modules."""
    try:
        from . import face_detection, face_recognition, face_analysis
        return {
            'face_detection': face_detection,
            'face_recognition': face_recognition,
            'face_analysis': face_analysis
        }
    except ImportError as e:
        logger.error(f"Could not import individual endpoints: {e}")
        return None

def create_main_router() -> Optional[APIRouter]:
    """
    Create main API router with all endpoints.
    
    Returns:
        APIRouter instance or None if creation failed
    """
    try:
        main_router = APIRouter()
        
        # Try to get complete endpoints first
        complete_endpoints = get_complete_endpoints()
        if complete_endpoints:
            # Use complete endpoints (recommended approach)
            main_router.include_router(
                complete_endpoints['face_detection_router'],
                prefix="/api",
                tags=["Face Detection"]
            )
            main_router.include_router(
                complete_endpoints['face_recognition_router'], 
                prefix="/api",
                tags=["Face Recognition"]
            )
            main_router.include_router(
                complete_endpoints['face_analysis_router'],
                prefix="/api", 
                tags=["Face Analysis"]
            )
            logger.info("Main router created with complete endpoints")
            
        else:
            # Fallback to individual endpoints
            individual_endpoints = get_individual_endpoints()
            if individual_endpoints:
                if hasattr(individual_endpoints['face_detection'], 'router'):
                    main_router.include_router(
                        individual_endpoints['face_detection'].router,
                        prefix="/api/face-detection",
                        tags=["Face Detection"]
                    )
                if hasattr(individual_endpoints['face_recognition'], 'router'):
                    main_router.include_router(
                        individual_endpoints['face_recognition'].router,
                        prefix="/api/face-recognition", 
                        tags=["Face Recognition"]
                    )
                if hasattr(individual_endpoints['face_analysis'], 'router'):
                    main_router.include_router(
                        individual_endpoints['face_analysis'].router,
                        prefix="/api/face-analysis",
                        tags=["Face Analysis"] 
                    )
                logger.info("Main router created with individual endpoints")
            else:
                logger.error("No endpoints available for router creation")
                return None
        
        return main_router
        
    except Exception as e:
        logger.error(f"Failed to create main router: {e}")
        return None

def get_api_documentation() -> Dict[str, Any]:
    """
    Get API documentation structure.
    
    Returns:
        API documentation dictionary
    """
    return {
        "info": API_CONFIG,
        "groups": API_GROUPS,
        "image_config": IMAGE_CONFIG,
        "endpoints": {
            "face_detection": [
                "POST /api/face-detection/detect",
                "POST /api/face-detection/detect-base64", 
                "POST /api/face-detection/detect-batch",
                "GET /api/face-detection/health",
                "GET /api/face-detection/models/status"
            ],
            "face_recognition": [
                "POST /api/face-recognition/add-face",
                "POST /api/face-recognition/add-face-json",
                "POST /api/face-recognition/extract-embedding",
                "POST /api/face-recognition/recognize",
                "POST /api/face-recognition/compare",
                "GET /api/face-recognition/gallery/get",
                "GET /api/face-recognition/health"
            ],
            "face_analysis": [
                "POST /api/face-analysis/analyze",
                "POST /api/face-analysis/analyze-json", 
                "POST /api/face-analysis/batch",
                "GET /api/face-analysis/health"
            ],
            "system": [
                "GET /health",
                "GET /",
                "GET /docs",
                "GET /redoc"
            ]
        }
    }

def validate_image_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate image configuration parameters.
    
    Args:
        config: Image configuration to validate
        
    Returns:
        Validated configuration
    """
    validated_config = IMAGE_CONFIG.copy()
    validated_config.update(config)
    
    # Validate file size (1MB to 50MB)
    validated_config["max_file_size"] = max(
        1024 * 1024,  # 1MB minimum
        min(50 * 1024 * 1024, validated_config.get("max_file_size", IMAGE_CONFIG["max_file_size"]))  # 50MB maximum
    )
    
    # Validate batch size (1 to 50)
    validated_config["max_batch_size"] = max(
        1,
        min(50, validated_config.get("max_batch_size", IMAGE_CONFIG["max_batch_size"]))
    )
    
    # Validate image dimensions
    validated_config["max_image_dimension"] = max(
        512,
        min(8192, validated_config.get("max_image_dimension", IMAGE_CONFIG["max_image_dimension"]))
    )
    validated_config["min_image_dimension"] = max(
        16,
        min(256, validated_config.get("min_image_dimension", IMAGE_CONFIG["min_image_dimension"]))
    )
    
    return validated_config

def get_error_responses() -> Dict[str, Dict[str, Any]]:
    """
    Get standard error response definitions.
    
    Returns:
        Error response definitions
    """
    return {
        "400": {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"},
                            "timestamp": {"type": "string"}
                        }
                    }
                }
            }
        },
        "422": {
            "description": "Validation Error", 
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "loc": {"type": "array"},
                                        "msg": {"type": "string"},
                                        "type": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object", 
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"},
                            "timestamp": {"type": "string"}
                        }
                    }
                }
            }
        },
        "503": {
            "description": "Service Unavailable",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "service": {"type": "string"},
                            "status": {"type": "string"}
                        }
                    }
                }
            }
        }
    }

def check_api_health() -> Dict[str, Any]:
    """
    Check API package health and availability.
    
    Returns:
        Health status dictionary
    """
    health_status = {
        "package_version": __version__,
        "endpoints_available": {},
        "modules_status": {},
        "overall_status": "unknown"
    }
    
    try:
        # Check complete endpoints
        complete_endpoints = get_complete_endpoints()
        health_status["endpoints_available"]["complete_endpoints"] = complete_endpoints is not None
        
        # Check individual endpoints  
        individual_endpoints = get_individual_endpoints()
        health_status["endpoints_available"]["individual_endpoints"] = individual_endpoints is not None
        
        # Check specific modules
        modules_to_check = ["face_detection", "face_recognition", "face_analysis"]
        for module_name in modules_to_check:
            try:
                if individual_endpoints and module_name in individual_endpoints:
                    health_status["modules_status"][module_name] = True
                else:
                    # Try direct import
                    __import__(f".{module_name}", package=__name__)
                    health_status["modules_status"][module_name] = True
            except ImportError:
                health_status["modules_status"][module_name] = False
        
        # Determine overall status
        if complete_endpoints or individual_endpoints:
            available_modules = sum(health_status["modules_status"].values())
            total_modules = len(health_status["modules_status"])
            
            if available_modules == total_modules:
                health_status["overall_status"] = "healthy"
            elif available_modules > 0:
                health_status["overall_status"] = "degraded"
            else:
                health_status["overall_status"] = "unhealthy"
        else:
            health_status["overall_status"] = "unhealthy"
            
    except Exception as e:
        logger.error(f"Error checking API health: {e}")
        health_status["overall_status"] = "error"
        health_status["error"] = str(e)
    
    return health_status

def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Package information dictionary
    """
    return {
        "version": __version__,
        "package_name": __package_name__,
        "api_config": API_CONFIG,
        "api_groups": API_GROUPS,
        "image_config": IMAGE_CONFIG,
        "documentation": get_api_documentation(),
        "health_status": check_api_health(),
        "error_responses": get_error_responses()
    }

# Export public interface
__all__ = [
    '__version__',
    '__package_name__',
    'API_CONFIG',
    'API_GROUPS', 
    'IMAGE_CONFIG',
    'get_complete_endpoints',
    'get_individual_endpoints',
    'create_main_router',
    'get_api_documentation',
    'validate_image_config',
    'get_error_responses',
    'check_api_health',
    'get_package_info'
]

# Package initialization
logger.info(f"API package v{__version__} loaded")

# Check API health on import
try:
    health = check_api_health()
    logger.info(f"API health status: {health['overall_status']}")
    
    if health["overall_status"] == "degraded":
        missing_modules = [name for name, available in health["modules_status"].items() if not available]
        logger.warning(f"Some API modules unavailable: {missing_modules}")
    elif health["overall_status"] == "unhealthy":
        logger.error("API package is unhealthy - no endpoints available")
        
except Exception as e:
    logger.warning(f"Failed to check API health: {e}")