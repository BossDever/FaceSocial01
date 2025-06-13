"""
AI Services Package for Face Recognition System
Contains all AI-related services: detection, recognition, and analysis
"""

import logging
from typing import Dict, Any, Optional, Type, Union
import asyncio

# Package metadata
__version__ = "2.0.0"
__package_name__ = "ai_services"

# Setup logging for this package
logger = logging.getLogger(__name__)

# Service registry for managing service instances
_service_registry: Dict[str, Any] = {}

# Service status tracking
_service_status: Dict[str, Dict[str, Any]] = {}

def register_service(name: str, service_instance: Any) -> None:
    """
    Register a service instance in the global registry.
    
    Args:
        name: Service name
        service_instance: Service instance
    """
    global _service_registry
    _service_registry[name] = service_instance
    _service_status[name] = {
        "registered": True,
        "initialized": hasattr(service_instance, 'initialize'),
        "type": type(service_instance).__name__
    }
    logger.info(f"Service '{name}' registered successfully")

def get_service(name: str) -> Optional[Any]:
    """
    Get a service instance from the registry.
    
    Args:
        name: Service name
        
    Returns:
        Service instance or None if not found
    """
    return _service_registry.get(name)

def list_services() -> Dict[str, Dict[str, Any]]:
    """
    List all registered services and their status.
    
    Returns:
        Dictionary of service information
    """
    return _service_status.copy()

def unregister_service(name: str) -> bool:
    """
    Unregister a service from the registry.
    
    Args:
        name: Service name
        
    Returns:
        True if service was found and removed, False otherwise
    """
    global _service_registry, _service_status
    
    if name in _service_registry:
        del _service_registry[name]
        del _service_status[name]
        logger.info(f"Service '{name}' unregistered successfully")
        return True
    
    logger.warning(f"Service '{name}' not found for unregistration")
    return False

async def shutdown_all_services() -> None:
    """
    Shutdown all registered services that have a shutdown method.
    """
    logger.info("Shutting down all AI services...")
    
    shutdown_tasks = []
    for name, service in _service_registry.items():
        if hasattr(service, 'shutdown') and callable(service.shutdown):
            logger.info(f"Shutting down service: {name}")
            try:
                if asyncio.iscoroutinefunction(service.shutdown):
                    shutdown_tasks.append(service.shutdown())
                else:
                    service.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
    
    # Wait for all async shutdowns to complete
    if shutdown_tasks:
        try:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during async service shutdown: {e}")
    
    # Clear registry
    _service_registry.clear()
    _service_status.clear()
    
    logger.info("All AI services shutdown completed")

# Lazy imports to avoid circular dependencies
def get_face_detection_service():
    """Lazy import for face detection service."""
    try:
        from .face_detection.face_detection_service import FaceDetectionService
        return FaceDetectionService
    except ImportError as e:
        logger.error(f"Could not import FaceDetectionService: {e}")
        return None

def get_face_recognition_service():
    """Lazy import for face recognition service."""
    try:
        from .face_recognition.face_recognition_service import FaceRecognitionService
        return FaceRecognitionService
    except ImportError as e:
        logger.error(f"Could not import FaceRecognitionService: {e}")
        return None

def get_face_analysis_service():
    """Lazy import for face analysis service."""
    try:
        from .face_analysis.face_analysis_service import FaceAnalysisService
        return FaceAnalysisService
    except ImportError as e:
        logger.error(f"Could not import FaceAnalysisService: {e}")
        return None

def get_vram_manager():
    """Lazy import for VRAM manager."""
    try:
        from .common.vram_manager import VRAMManager
        return VRAMManager
    except ImportError as e:
        logger.error(f"Could not import VRAMManager: {e}")
        return None

# Service factory function
def create_service(service_type: str, vram_manager: Any = None, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    Factory function to create service instances.
    
    Args:
        service_type: Type of service ('detection', 'recognition', 'analysis')
        vram_manager: VRAM manager instance
        config: Service configuration
        
    Returns:
        Service instance or None if creation failed
    """
    service_classes = {
        'detection': get_face_detection_service(),
        'recognition': get_face_recognition_service(),
        'analysis': get_face_analysis_service()
    }
    
    service_class = service_classes.get(service_type)
    if service_class is None:
        logger.error(f"Unknown service type: {service_type}")
        return None
    
    try:
        # Create service instance
        if service_type == 'analysis':
            # Analysis service has different constructor signature
            service = service_class(
                vram_manager=vram_manager,
                config=config,
                face_detection_service=None,  # Will be set later
                face_recognition_service=None  # Will be set later
            )
        else:
            service = service_class(vram_manager=vram_manager, config=config)
        
        logger.info(f"Created {service_type} service successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to create {service_type} service: {e}")
        return None

# Package initialization check
def check_package_health() -> Dict[str, Any]:
    """
    Check the health of the AI services package.
    
    Returns:
        Dictionary with health information
    """
    health_info = {
        "package_version": __version__,
        "services_registered": len(_service_registry),
        "service_list": list(_service_registry.keys()),
        "import_status": {}
    }
    
    # Check import status for each service
    services_to_check = {
        'detection': get_face_detection_service,
        'recognition': get_face_recognition_service,
        'analysis': get_face_analysis_service,
        'vram_manager': get_vram_manager
    }
    
    for service_name, import_func in services_to_check.items():
        try:
            service_class = import_func()
            health_info["import_status"][service_name] = {
                "available": service_class is not None,
                "class_name": service_class.__name__ if service_class else None
            }
        except Exception as e:
            health_info["import_status"][service_name] = {
                "available": False,
                "error": str(e)
            }
    
    return health_info

# Export public interface
__all__ = [
    "__version__",
    "__package_name__",
    "register_service",
    "get_service", 
    "list_services",
    "unregister_service",
    "shutdown_all_services",
    "get_face_detection_service",
    "get_face_recognition_service", 
    "get_face_analysis_service",
    "get_vram_manager",
    "create_service",
    "check_package_health"
]

# Package-level initialization
logger.info(f"AI Services package v{__version__} loaded")

# Optional: Check package health on import
try:
    health = check_package_health()
    available_services = sum(1 for status in health["import_status"].values() if status["available"])
    total_services = len(health["import_status"])
    logger.info(f"Package health check: {available_services}/{total_services} services available")
except Exception as e:
    logger.warning(f"Package health check failed: {e}")