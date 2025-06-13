"""
Common utilities and shared components for AI Services
Contains VRAM management, statistics, and utility functions
"""

import logging
from typing import Dict, Any, Optional, Union, List

# Package metadata
__version__ = "2.0.0"
__package_name__ = "ai_services.common"

# Setup logging
logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies and handle missing dependencies
def get_vram_manager():
    """Lazy import for VRAM Manager."""
    try:
        from .vram_manager import VRAMManager, AllocationPriority, AllocationLocation, ModelAllocation
        return {
            'VRAMManager': VRAMManager,
            'AllocationPriority': AllocationPriority,
            'AllocationLocation': AllocationLocation,
            'ModelAllocation': ModelAllocation
        }
    except ImportError as e:
        logger.error(f"Could not import VRAM Manager: {e}")
        return None

def get_stats_module():
    """Lazy import for Statistics module."""
    try:
        from .stats import (
            ModelPerformanceStats, 
            ServiceStats, 
            get_service_stats, 
            get_all_stats, 
            reset_all_stats,
            cleanup_stats
        )
        return {
            'ModelPerformanceStats': ModelPerformanceStats,
            'ServiceStats': ServiceStats,
            'get_service_stats': get_service_stats,
            'get_all_stats': get_all_stats,
            'reset_all_stats': reset_all_stats,
            'cleanup_stats': cleanup_stats
        }
    except ImportError as e:
        logger.error(f"Could not import Statistics module: {e}")
        return None

def get_utils_module():
    """Lazy import for Utils module."""
    try:
        from .utils import (
            ONNX_AVAILABLE,
            TORCH_AVAILABLE, 
            OPENCV_AVAILABLE,
            NUMPY_AVAILABLE,
            ort,
            torch,
            cv2,
            np,
            check_dependencies,
            get_device_info,
            ensure_numpy_array,
            safe_import
        )
        return {
            'ONNX_AVAILABLE': ONNX_AVAILABLE,
            'TORCH_AVAILABLE': TORCH_AVAILABLE,
            'OPENCV_AVAILABLE': OPENCV_AVAILABLE,
            'NUMPY_AVAILABLE': NUMPY_AVAILABLE,
            'ort': ort,
            'torch': torch,
            'cv2': cv2,
            'np': np,
            'check_dependencies': check_dependencies,
            'get_device_info': get_device_info,
            'ensure_numpy_array': ensure_numpy_array,
            'safe_import': safe_import
        }
    except ImportError as e:
        logger.error(f"Could not import Utils module: {e}")
        return None

# Global registry for tracking common services
_common_services: Dict[str, Any] = {}

def register_common_service(name: str, service: Any) -> None:
    """
    Register a common service (like VRAM Manager).
    
    Args:
        name: Service name
        service: Service instance
    """
    global _common_services
    _common_services[name] = service
    logger.info(f"Common service '{name}' registered")

def get_common_service(name: str) -> Optional[Any]:
    """
    Get a registered common service.
    
    Args:
        name: Service name
        
    Returns:
        Service instance or None if not found
    """
    return _common_services.get(name)

def list_common_services() -> List[str]:
    """
    List all registered common services.
    
    Returns:
        List of service names
    """
    return list(_common_services.keys())

def clear_common_services() -> None:
    """Clear all registered common services."""
    global _common_services
    _common_services.clear()
    logger.info("All common services cleared")

# Utility functions for checking module availability
def check_module_availability() -> Dict[str, bool]:
    """
    Check availability of all common modules.
    
    Returns:
        Dictionary with module availability status
    """
    modules = {
        'vram_manager': get_vram_manager() is not None,
        'stats': get_stats_module() is not None,
        'utils': get_utils_module() is not None
    }
    
    return modules

def get_system_requirements() -> Dict[str, Any]:
    """
    Get system requirements and availability.
    
    Returns:
        Dictionary with system information
    """
    utils = get_utils_module()
    if utils:
        return {
            'dependencies': utils['check_dependencies'](),
            'device_info': utils['get_device_info'](),
            'modules': check_module_availability()
        }
    else:
        return {
            'dependencies': {},
            'device_info': {},
            'modules': check_module_availability(),
            'error': 'Utils module not available'
        }

def initialize_common_services(config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    Initialize common services with provided configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with initialization results
    """
    results = {}
    
    # Initialize VRAM Manager
    vram_components = get_vram_manager()
    if vram_components and config:
        try:
            vram_config = config.get('vram_config', {})
            vram_manager = vram_components['VRAMManager'](vram_config)
            register_common_service('vram_manager', vram_manager)
            results['vram_manager'] = True
            logger.info("VRAM Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VRAM Manager: {e}")
            results['vram_manager'] = False
    else:
        results['vram_manager'] = False
    
    # Initialize Stats (no special initialization needed)
    stats_module = get_stats_module()
    if stats_module:
        results['stats'] = True
        logger.info("Stats module available")
    else:
        results['stats'] = False
    
    # Check Utils (no special initialization needed)
    utils_module = get_utils_module()
    if utils_module:
        results['utils'] = True
        logger.info("Utils module available")
    else:
        results['utils'] = False
    
    return results

async def shutdown_common_services() -> None:
    """Shutdown all common services."""
    logger.info("Shutting down common services...")
    
    # Shutdown VRAM Manager if available
    vram_manager = get_common_service('vram_manager')
    if vram_manager and hasattr(vram_manager, 'shutdown'):
        try:
            await vram_manager.shutdown()
            logger.info("VRAM Manager shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down VRAM Manager: {e}")
    
    # Cleanup stats
    stats_module = get_stats_module()
    if stats_module and 'cleanup_stats' in stats_module:
        try:
            stats_module['cleanup_stats']()
            logger.info("Stats cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up stats: {e}")
    
    # Clear services
    clear_common_services()
    logger.info("Common services shutdown completed")

def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Dictionary with package information
    """
    return {
        'version': __version__,
        'package_name': __package_name__,
        'available_modules': check_module_availability(),
        'registered_services': list_common_services(),
        'system_requirements': get_system_requirements()
    }

# Export public interface
__all__ = [
    '__version__',
    '__package_name__',
    'get_vram_manager',
    'get_stats_module',
    'get_utils_module',
    'register_common_service',
    'get_common_service',
    'list_common_services',
    'clear_common_services',
    'check_module_availability',
    'get_system_requirements',
    'initialize_common_services',
    'shutdown_common_services',
    'get_package_info'
]

# Package initialization
logger.info(f"AI Services Common package v{__version__} loaded")

# Check module availability on import
try:
    availability = check_module_availability()
    available_count = sum(availability.values())
    total_count = len(availability)
    logger.info(f"Module availability: {available_count}/{total_count} modules available")
    
    if available_count < total_count:
        missing_modules = [name for name, available in availability.items() if not available]
        logger.warning(f"Missing modules: {missing_modules}")
        
except Exception as e:
    logger.warning(f"Failed to check module availability: {e}")